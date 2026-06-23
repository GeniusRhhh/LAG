"""
编队引导模块 - 算法2.6精确实现
编队级引导目标与航向分配

严格按照报告中算法2.6的步骤实现：
1. 计算编队几何中心 p_formation
2. 计算编队中心→敌方中心的向量与直接航向 θ_direct
3. 对每个编队对(lead, wing)：
   - 计算编队对中心 → 敌方中心的单位方向
   - 长机目标点 = 敌方中心 - (R_target/2) * 方向 (近侧)
   - 僚机目标点 = 敌方中心 + (R_target/2) * 方向 (远侧)
4. 各机目标航向 = atan2(目标点 - 当前位置)
"""
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class GuidanceTarget:
    """引导目标"""
    fighter_id: str
    target_point: Tuple[float, float]  # (x, y) km
    target_heading: float  # 目标航向(度)，真北=0°，顺时针正
    distance_to_target: float  # 距离(km)


class FormationGuidance:
    """编队引导管理器 - 算法2.6精确实现
    
    核心逻辑（报告步骤1-10）：
    1. 编队几何中心 p_formation = (1/4)Σp_i
    2. Δp = p_enemy - p_formation; θ_direct = atan2(Δp_x, Δp_y)
    3. 对每个编队对: 
       - 长机* = p_enemy - (R_target/2)*e_pair  (在目标区域边界近侧)
       - 僚机* = p_enemy + (R_target/2)*e_pair  (在目标区域边界远侧)
    4. θ_i* = atan2(p_i* - p_i)
    """
    
    def __init__(
        self,
        formation_pairs: List[Tuple[str, str]] = None,
        bracket_far_km: Optional[float] = None,
        bracket_full_km: Optional[float] = None,
        half_r_max_km: Optional[float] = None,
        half_r_ratio: Optional[float] = None,
    ):
        """
        Args:
            formation_pairs: 编队配对 [(长机1, 僚机1), (长机2, 僚机2)]
        """
        self.formation_pairs = formation_pairs or [
            ('A0100', 'A0200'),  # 左编队
            ('A0300', 'A0400'),  # 右编队
        ]

        # 工程增强（不改变接口）：
        # 远距离阶段不直接给僚机分配“敌后远侧点”，否则会出现数百公里的绕路弧线。
        # 采用“渐进包夹”：
        # - dist >= bracket_far_km : offset=0，长/僚机都指向敌方中心（直拦截）
        # - dist <= bracket_full_km: offset=clamped(R_target/2)，恢复算法2.6近侧/远侧
        # - 中间线性插值。
        # 同时对 R_target/2 做与距离相关的上限钳制，避免不确定性导致目标点跳到几百公里外。
        def _env_float(name: str, default: float) -> float:
            raw = os.environ.get(name, '').strip()
            if not raw:
                return float(default)
            try:
                return float(raw)
            except Exception:
                return float(default)

        self.bracket_far_km = _env_float('CAP_FORMATION_BRACKET_FAR_KM', bracket_far_km if bracket_far_km is not None else 220.0)
        self.bracket_full_km = _env_float('CAP_FORMATION_BRACKET_FULL_KM', bracket_full_km if bracket_full_km is not None else 140.0)
        self.half_r_max_km = _env_float('CAP_FORMATION_HALF_R_MAX_KM', half_r_max_km if half_r_max_km is not None else 120.0)
        self.half_r_ratio = _env_float('CAP_FORMATION_HALF_R_RATIO', half_r_ratio if half_r_ratio is not None else 0.45)

        # 容错：确保阈值关系有效
        if self.bracket_full_km > self.bracket_far_km:
            self.bracket_full_km, self.bracket_far_km = self.bracket_far_km, self.bracket_full_km
        self.half_r_max_km = max(0.0, float(self.half_r_max_km))
        self.half_r_ratio = max(0.0, float(self.half_r_ratio))
        self._last_guidance: Dict[str, GuidanceTarget] = {}
    
    def compute_guidance(
        self,
        enemy_center: Tuple[float, float],
        confidence_radius: float,
        fighter_positions: Dict[str, Tuple[float, float]],
        responsibility_map: Optional[Dict[str, float]] = None,
        responsibility_role_map: Optional[Dict[str, str]] = None,
        responsibility_weight_scale: float = 1.0,
    ) -> Dict[str, GuidanceTarget]:
        """算法2.6：编队级引导目标与航向分配
        
        Args:
            enemy_center: 敌方估计中心 p_enemy (x, y) km
            confidence_radius: 置信区域半径 R_target (km)
            fighter_positions: 各机当前位置 {id: (x, y)}
            responsibility_map: 责任方向分配 {fighter_id: bearing_deg}
            responsibility_role_map: 责任角色 {fighter_id: primary/support}
        
        Returns:
            各机引导目标 {fighter_id: GuidanceTarget}
        """
        if not fighter_positions:
            return {}
        
        # ===== 步骤1: 编队几何中心 =====
        all_pos = list(fighter_positions.values())
        p_formation = (
            np.mean([p[0] for p in all_pos]),
            np.mean([p[1] for p in all_pos])
        )
        
        # ===== 步骤2: 编队中心→敌方中心的直接航向 =====
        delta_x = enemy_center[0] - p_formation[0]
        delta_y = enemy_center[1] - p_formation[1]
        # θ_direct = atan2(Δp_x, Δp_y)  (真北=0°, 顺时针正)
        # 此处未直接使用，但下面每个编队对会单独计算方向
        
        guidance = {}
        R = float(confidence_radius)  # R_target
        
        # ===== 步骤3-6: 对每个编队对分配长机/僚机目标点 =====
        for lead_id, wing_id in self.formation_pairs:
            # 检查飞机是否存在
            lead_pos = fighter_positions.get(lead_id)
            wing_pos = fighter_positions.get(wing_id)
            
            if lead_pos is None and wing_pos is None:
                continue
            
            # 步骤4: 编队对中心
            if lead_pos and wing_pos:
                p_pair = ((lead_pos[0] + wing_pos[0]) / 2, (lead_pos[1] + wing_pos[1]) / 2)
            elif lead_pos:
                p_pair = lead_pos
            else:
                p_pair = wing_pos
            
            # 步骤5: 编队对中心→敌方中心的单位向量 e_pair
            dp_x = enemy_center[0] - p_pair[0]
            dp_y = enemy_center[1] - p_pair[1]
            dist_to_enemy = np.sqrt(dp_x**2 + dp_y**2)
            if dist_to_enemy < 1.0:
                dist_to_enemy = 1.0
            e_x = dp_x / dist_to_enemy
            e_y = dp_y / dist_to_enemy
            
            # 工程增强：渐进包夹 + half_R 钳制
            # - 避免在远距离阶段把僚机推到“敌后远侧点”导致绕远
            # - 避免 R_target 过大时目标点跳变
            half_R = R / 2.0
            half_R_cap = min(
                self.half_r_max_km if self.half_r_max_km > 0 else half_R,
                (dist_to_enemy * self.half_r_ratio) if self.half_r_ratio > 0 else half_R,
            )
            half_R_eff = min(half_R, half_R_cap) if half_R_cap > 0 else half_R

            if dist_to_enemy >= self.bracket_far_km:
                w = 0.0
            elif dist_to_enemy <= self.bracket_full_km:
                w = 1.0
            else:
                denom = max(1e-6, (self.bracket_far_km - self.bracket_full_km))
                w = (self.bracket_far_km - dist_to_enemy) / denom
                w = float(np.clip(w, 0.0, 1.0))

            offset = w * half_R_eff
            
            if lead_pos:
                lead_target = (
                    enemy_center[0] - offset * e_x,
                    enemy_center[1] - offset * e_y
                )
                guidance[lead_id] = self._make_guidance(lead_id, lead_pos, lead_target)
            
            if wing_pos:
                wing_target = (
                    enemy_center[0] + offset * e_x,
                    enemy_center[1] + offset * e_y
                )
                guidance[wing_id] = self._make_guidance(wing_id, wing_pos, wing_target)
        
        if responsibility_map:
            guidance = self._apply_responsibility_guidance(
                guidance=guidance,
                enemy_center=enemy_center,
                fighter_positions=fighter_positions,
                responsibility_map=responsibility_map,
                responsibility_role_map=responsibility_role_map or {},
                responsibility_weight_scale=responsibility_weight_scale,
            )

        self._last_guidance = guidance
        return guidance

    def _apply_responsibility_guidance(
        self,
        guidance: Dict[str, GuidanceTarget],
        enemy_center: Tuple[float, float],
        fighter_positions: Dict[str, Tuple[float, float]],
        responsibility_map: Dict[str, float],
        responsibility_role_map: Dict[str, str],
        responsibility_weight_scale: float,
    ) -> Dict[str, GuidanceTarget]:
        """将任务分配层输出直接映射到引导层目标点。"""
        blend_scale = float(np.clip(responsibility_weight_scale, 0.0, 1.0))
        shaped_guidance: Dict[str, GuidanceTarget] = {}
        for fighter_id, base_guidance in guidance.items():
            current_pos = fighter_positions.get(fighter_id)
            responsibility_bearing = responsibility_map.get(fighter_id)
            if current_pos is None or responsibility_bearing is None:
                shaped_guidance[fighter_id] = base_guidance
                continue

            role = str(responsibility_role_map.get(fighter_id, "primary"))
            dist_to_enemy = float(np.hypot(enemy_center[0] - current_pos[0], enemy_center[1] - current_pos[1]))
            if dist_to_enemy <= 60.0:
                shaped_guidance[fighter_id] = base_guidance
                continue

            if role == "primary":
                if dist_to_enemy >= 220.0:
                    weight = 0.90
                elif dist_to_enemy <= 120.0:
                    weight = 0.20
                else:
                    weight = 0.20 + 0.70 * (dist_to_enemy - 120.0) / 100.0
                lookahead = float(np.clip(dist_to_enemy, 100.0, 260.0))
            else:
                if dist_to_enemy >= 220.0:
                    weight = 0.55
                elif dist_to_enemy <= 140.0:
                    weight = 0.05
                else:
                    weight = 0.05 + 0.50 * (dist_to_enemy - 140.0) / 80.0
                lookahead = float(np.clip(dist_to_enemy * 0.80, 80.0, 180.0))

            weight *= blend_scale
            responsibility_rad = np.radians(float(responsibility_bearing))
            responsibility_point = (
                float(current_pos[0] + np.sin(responsibility_rad) * lookahead),
                float(current_pos[1] + np.cos(responsibility_rad) * lookahead),
            )
            target_point = (
                float((1.0 - weight) * float(base_guidance.target_point[0]) + weight * responsibility_point[0]),
                float((1.0 - weight) * float(base_guidance.target_point[1]) + weight * responsibility_point[1]),
            )
            shaped_guidance[fighter_id] = self._make_guidance(fighter_id, current_pos, target_point)

        return shaped_guidance
    
    def _make_guidance(
        self, fighter_id: str, 
        current_pos: Tuple[float, float], 
        target_point: Tuple[float, float]
    ) -> GuidanceTarget:
        """步骤7-8: 由目标点反算目标航向
        
        θ_i* = atan2(Δp_i_x, Δp_i_y)
        """
        dx = target_point[0] - current_pos[0]
        dy = target_point[1] - current_pos[1]
        # atan2(dx, dy) 给出真北=0°，顺时针正
        heading = np.degrees(np.arctan2(dx, dy)) % 360
        distance = np.sqrt(dx**2 + dy**2)
        
        return GuidanceTarget(
            fighter_id=fighter_id,
            target_point=target_point,
            target_heading=heading,
            distance_to_target=distance
        )
    
    def get_guidance(self, fighter_id: str) -> Optional[GuidanceTarget]:
        return self._last_guidance.get(fighter_id)
    
    def get_all_guidance(self) -> Dict[str, GuidanceTarget]:
        return self._last_guidance.copy()
