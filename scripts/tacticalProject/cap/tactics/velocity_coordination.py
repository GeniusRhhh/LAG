"""
速度协调模块 - 算法2.7精确实现
确保编队同步到达目标区域

严格按照报告中算法2.7的步骤实现：
- 步骤1-3：计算各机当前ETA
- 步骤4-34：对每个编队对做同步速度决策
  - 已同步：保持速度
  - 未同步：构造候选A（先到者减速）和候选B（后到者加速），取代价小者
- 代价函数：J = |T_lead - T_wing| + λ * Σ(v_i - v_nominal)^2
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpeedCommand:
    """速度命令"""
    fighter_id: str
    target_speed: float  # 目标速度 (m/s)
    speed_index: int  # 速度命令索引 (0-7 for discrete control)
    eta: float  # 预计到达时间 (s)


class MissionPhase:
    """任务阶段枚举"""
    PATROL = "PATROL"
    INTERCEPT = "INTERCEPT"
    ENGAGE = "ENGAGE"
    EVADE = "EVADE"
    RTB = "RTB"


class VelocityCoordination:
    """速度协调管理器 - 算法2.7精确实现（V2：阶段感知）
    
    对每个编队对(长机-僚机)：
    1. 计算各机当前速度下的ETA
    2. 判断|T_lead - T_wing| ≤ ΔT_sync？
       - 是：保持当前速度
       - 否：构造候选A(先到者减速) + 候选B(后到者加速)
    3. 按代价函数J = |ΔT| + λ·Σ(v-v_nominal)² 取较小者
    
    🔥 V2改进：根据任务阶段动态调整速度约束
    - PATROL: 200-250 m/s（节省燃料）
    - INTERCEPT: 250-300 m/s（快速前出）
    - ENGAGE: 250-330 m/s（最大机动性）
    """
    
    # 速度档位 (m/s)
    SPEED_LEVELS = [150, 180, 200, 220, 250, 280, 300, 330]
    
    # 阶段相关速度约束 (m/s)
    PHASE_SPEED_CONSTRAINTS = {
        MissionPhase.PATROL: {'v_min': 200.0, 'v_max': 250.0, 'v_nominal': 220.0},
        MissionPhase.INTERCEPT: {'v_min': 250.0, 'v_max': 300.0, 'v_nominal': 280.0},
        MissionPhase.ENGAGE: {'v_min': 250.0, 'v_max': 330.0, 'v_nominal': 280.0},
        MissionPhase.EVADE: {'v_min': 280.0, 'v_max': 330.0, 'v_nominal': 300.0},
        MissionPhase.RTB: {'v_min': 220.0, 'v_max': 280.0, 'v_nominal': 250.0},
    }
    
    # 默认约束（向后兼容）
    V_MIN = 150.0   # m/s
    V_MAX = 330.0    # m/s
    V_NOMINAL = 250.0  # 标称速度 m/s
    DELTA_T_SYNC = 15.0  # 同步到达容差 (s) - 报告中ΔT_sync
    LAMBDA = 0.001  # 代价函数中速度偏差权重λ
    
    def __init__(self, formation_pairs: List[Tuple[str, str]] = None):
        self.formation_pairs = formation_pairs or [
            ('A0100', 'A0200'),
            ('A0300', 'A0400'),
        ]
        self._last_commands: Dict[str, SpeedCommand] = {}
        self._current_phase = MissionPhase.PATROL  # 默认巡逻阶段
    
    def set_mission_phase(self, phase: str):
        """设置当前任务阶段
        
        Args:
            phase: 任务阶段 (PATROL, INTERCEPT, ENGAGE, EVADE, RTB)
        """
        self._current_phase = phase
    
    def _get_phase_constraints(self) -> dict:
        """获取当前阶段的速度约束"""
        return self.PHASE_SPEED_CONSTRAINTS.get(
            self._current_phase,
            {'v_min': self.V_MIN, 'v_max': self.V_MAX, 'v_nominal': self.V_NOMINAL}
        )
    
    def compute_coordinated_speeds(
        self,
        fighter_positions: Dict[str, Tuple[float, float]],
        fighter_speeds: Dict[str, float],
        target_positions: Dict[str, Tuple[float, float]],
        mission_phase: str = None  # 🔥 V2新增：任务阶段参数
    ) -> Dict[str, SpeedCommand]:
        """算法2.7：速度协调优化（V3：强制阶段约束）
        
        Args:
            fighter_positions: 各机当前位置 {id: (x, y)} km
            fighter_speeds: 各机当前速度 {id: speed} m/s
            target_positions: 各机引导目标位置 {id: (x, y)} km
            mission_phase: 任务阶段 (PATROL, INTERCEPT, ENGAGE等)，可选
        
        Returns:
            速度命令 {fighter_id: SpeedCommand}
        """
        # 🔥 V2：更新任务阶段
        if mission_phase is not None:
            self.set_mission_phase(mission_phase)
        
        # 获取阶段约束
        constraints = self._get_phase_constraints()
        v_min = constraints['v_min']
        v_max = constraints['v_max']
        v_nominal = constraints['v_nominal']
        
        if not fighter_positions or not target_positions:
            return {}
        
        # ===== 步骤1-3：计算各机距离和ETA =====
        distances = {}  # m
        etas = {}       # s
        for fid, pos in fighter_positions.items():
            if fid not in target_positions:
                continue
            tgt = target_positions[fid]
            dist = np.sqrt((tgt[0] - pos[0])**2 + (tgt[1] - pos[1])**2) * 1000.0  # km -> m
            speed = fighter_speeds.get(fid, v_nominal)
            # 🔥 V3：强制速度在阶段约束范围内
            speed = np.clip(speed, v_min, v_max)
            distances[fid] = dist
            etas[fid] = dist / speed  # T_i^cur = d_i / v_i^current
        
        if not etas:
            return {}
        
        # ===== 步骤4-35：对每个编队对做同步速度决策 =====
        commands = {}
        for lead_id, wing_id in self.formation_pairs:
            # 确认双方都有数据
            if lead_id not in etas or wing_id not in etas:
                for fid in [lead_id, wing_id]:
                    if fid in etas:
                        # 🔥 V3：使用阶段标称速度作为默认值
                        commands[fid] = self._make_command(fid, v_nominal, etas[fid])
                continue
            
            T_lead_cur = etas[lead_id]
            T_wing_cur = etas[wing_id]
            d_lead = distances[lead_id]
            d_wing = distances[wing_id]
            v_lead_cur = np.clip(fighter_speeds.get(lead_id, v_nominal), v_min, v_max)
            v_wing_cur = np.clip(fighter_speeds.get(wing_id, v_nominal), v_min, v_max)
            
            # ===== 步骤5-7：判断是否已同步 =====
            if abs(T_lead_cur - T_wing_cur) <= self.DELTA_T_SYNC:
                # 🔥 V3：已同步时，确保速度不低于阶段最小值
                v_lead_sync = max(v_lead_cur, v_min)
                v_wing_sync = max(v_wing_cur, v_min)
                commands[lead_id] = self._make_command(lead_id, v_lead_sync, T_lead_cur)
                commands[wing_id] = self._make_command(wing_id, v_wing_sync, T_wing_cur)
                continue
            
            # ===== 步骤8-34：未同步，构造两候选并比较 =====
            
            # --- 候选A：先到者减速 (步骤9-17) ---
            if T_lead_cur < T_wing_cur:
                # 长机先到 -> 长机减速（但不低于v_min）
                T_lead_tar = max(T_lead_cur, T_wing_cur - self.DELTA_T_SYNC)
                v_lead_A = np.clip(d_lead / T_lead_tar, v_min, v_max) if T_lead_tar > 0 else v_nominal
                v_wing_A = v_wing_cur
            else:
                # 僚机先到 -> 僚机减速（但不低于v_min）
                T_wing_tar = max(T_wing_cur, T_lead_cur - self.DELTA_T_SYNC)
                v_wing_A = np.clip(d_wing / T_wing_tar, v_min, v_max) if T_wing_tar > 0 else v_nominal
                v_lead_A = v_lead_cur
            
            # --- 候选B：后到者加速 (步骤18-26) ---
            if T_lead_cur < T_wing_cur:
                # 长机先到 -> 僚机加速
                T_wing_tar = min(T_wing_cur, T_lead_cur + self.DELTA_T_SYNC)
                v_wing_B = np.clip(d_wing / T_wing_tar, v_min, v_max) if T_wing_tar > 0 else v_nominal
                v_lead_B = v_lead_cur
            else:
                # 僚机先到 -> 长机加速
                T_lead_tar = min(T_lead_cur, T_wing_cur + self.DELTA_T_SYNC)
                v_lead_B = np.clip(d_lead / T_lead_tar, v_min, v_max) if T_lead_tar > 0 else v_nominal
                v_wing_B = v_wing_cur
            
            # --- 步骤27-31：计算候选下的实际到达时间 ---
            T_lead_A = d_lead / v_lead_A if v_lead_A > 0 else 9999
            T_wing_A = d_wing / v_wing_A if v_wing_A > 0 else 9999
            T_lead_B = d_lead / v_lead_B if v_lead_B > 0 else 9999
            T_wing_B = d_wing / v_wing_B if v_wing_B > 0 else 9999
            
            # --- 步骤32-33：计算代价 J = |ΔT| + λ·Σ(v - v_nominal)² ---
            # 🔥 V2：使用阶段相关的v_nominal
            J_A = abs(T_lead_A - T_wing_A) + self.LAMBDA * (
                (v_lead_A - v_nominal)**2 + (v_wing_A - v_nominal)**2
            )
            J_B = abs(T_lead_B - T_wing_B) + self.LAMBDA * (
                (v_lead_B - v_nominal)**2 + (v_wing_B - v_nominal)**2
            )
            
            # --- 步骤34：选择代价较小者 ---
            if J_A <= J_B:
                v_lead_star, v_wing_star = v_lead_A, v_wing_A
                eta_sync = (T_lead_A + T_wing_A) / 2
            else:
                v_lead_star, v_wing_star = v_lead_B, v_wing_B
                eta_sync = (T_lead_B + T_wing_B) / 2
            
            # 🔥 V3：最终确保速度不低于阶段最小值
            v_lead_star = max(v_lead_star, v_min)
            v_wing_star = max(v_wing_star, v_min)
            
            commands[lead_id] = self._make_command(lead_id, v_lead_star, eta_sync)
            commands[wing_id] = self._make_command(wing_id, v_wing_star, eta_sync)
        
        self._last_commands = commands
        return commands
    
    def _make_command(self, fighter_id: str, target_speed: float, eta: float) -> SpeedCommand:
        """将连续速度映射到最近的离散档位"""
        speed_index = int(np.argmin([abs(s - target_speed) for s in self.SPEED_LEVELS]))
        return SpeedCommand(
            fighter_id=fighter_id,
            target_speed=self.SPEED_LEVELS[speed_index],
            speed_index=speed_index,
            eta=eta
        )
    
    def get_command(self, fighter_id: str) -> Optional[SpeedCommand]:
        return self._last_commands.get(fighter_id)
    
    def get_speed_index(self, fighter_id: str, default: int = 4) -> int:
        cmd = self._last_commands.get(fighter_id)
        return cmd.speed_index if cmd else default
