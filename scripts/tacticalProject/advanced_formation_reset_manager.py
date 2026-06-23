"""
高级编队重置管理器 - 解决编队重置时的位置冲突和姿态问题

功能：
1. 提供安全的编队重置位置计算
2. 确保僚机在长机附近的正确位置
3. 避免重置时的碰撞
4. 支持更复杂的编队构型
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from formation_reset_manager import FormationResetManager

class AdvancedFormationResetManager(FormationResetManager):
    """高级编队重置管理器 - 完整实现版"""
    
    def __init__(self):
        super().__init__()
        self.active = False
        self.start_time = 0.0
        self.phase = 'inactive'  # inactive, turning, regrouping
        self.target_heading = 0.0  # 北向
        self.formation_spacing = 9000.0  # 约5海里
        self.regroup_tolerance = 2000.0  # 2km误差
        self.heading_tolerance = 15.0    # 🔥 放宽容差到15度，满足“0±15°可直接进攻”
        self.agent_states = {}
        self.zero_heading_hold_start = None  # 保持0度航向的开始时间
        self.zero_heading_hold_duration = 1.0  # 🔥 优化：缩短保持时间到1秒，避免转圈
        self.completion_check_count = {}  # 跟踪每个agent的完成检查次数
        self.active_agents = []
        
    def start_formation_reset(self, env, current_time: float, agent_ids: Optional[List[str]] = None) -> bool:
        """启动编队重置流程"""
        self.active = True
        self.start_time = current_time
        self.phase = 'turning'
        self.agent_states = {}
        self.active_agents = list(agent_ids or [])
        try:
            from utils.trace_logger import trace_event
            trace_event(
                事件="编队重整-启动",
                env=env,
                模块="advanced_formation_reset_manager",
                类型="ACTION",
                状态="START",
                说明="进入编队重整流程：先对准北向0°(容差±15°)，再保持1秒收敛，避免转圈",
                数据={
                    "target_heading_deg": float(self.target_heading),
                    "heading_tolerance_deg": float(self.heading_tolerance),
                    "hold_s": float(self.zero_heading_hold_duration),
                    "spacing_m": float(self.formation_spacing),
                },
            )
        except Exception:
            pass
        return True
        
    def is_reset_active(self) -> bool:
        """检查是否正在进行编队重置"""
        return self.active
        
    def execute_formation_reset(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        执行编队重置机动
        🔥 核心逻辑：从当前航向（通常180°南向）转到0°（北向）
        长机朝左转（逆时针），僚机朝右转（顺时针），向内侧靠拢
        返回: (alt_cmd, hdg_cmd, vel_cmd)
        """
        if not self.active:
            return 7, 8, 3
            
        from envs.JSBSim.core.catalog import Catalog as c
        from tactical_utils import TacticalUtils
        
        aircraft = env.agents[agent_id]
        current_heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
        current_alt = aircraft.get_property_value(c.position_h_sl_m)
        
        target_heading = 0.0  # 目标：北向（0度）

        # 计算到目标航向的角度差（标准化到[-180, 180]）
        heading_diff = TacticalUtils.normalize_angle_diff(target_heading - current_heading)

        # 🔥 规则：若已在0±15°内，直接判定完成，不再转圈
        if abs(heading_diff) <= 15.0:
            self.agent_states[agent_id] = 'turned'
            hdg_cmd = 8  # 保持航向
            try:
                from utils.trace_logger import trace_if_changed
                trace_if_changed(
                    key=f"formation:turned:{agent_id}",
                    value=True,
                    标题="编队重整-到位",
                    env=env,
                    模块="advanced_formation_reset_manager",
                    类型="STATE",
                    状态="CHANGED",
                    我机=str(agent_id),
                    阶段="FORMATION_RESET",
                    战术="FORMATION_RESET",
                    说明="航向进入0±15°容差：标记为到位并保持航向",
                    数据={"heading_deg": float(current_heading), "diff_deg": float(heading_diff)},
                )
            except Exception:
                pass
        elif self.phase == 'holding_zero':
            # 保持阶段，强制保持航向
            hdg_cmd = 8
            try:
                from utils.trace_logger import trace_throttle
            except Exception:
                trace_throttle = None
            if trace_throttle is not None:
                try:
                    trace_throttle(
                        key=f"formation:holding:{agent_id}",
                        min_steps=150,
                        标题="编队重整-保持",
                        env=env,
                        模块="advanced_formation_reset_manager",
                        类型="ACTION",
                        状态="PERIODIC",
                        我机=str(agent_id),
                        阶段="FORMATION_RESET",
                        战术="FORMATION_RESET",
                        说明="保持阶段：强制保持0°航向以收敛",
                        数据={"heading_deg": float(current_heading), "diff_deg": float(heading_diff)},
                    )
                except Exception:
                    pass
        else:
            # 🔥 最短路径转向0度：heading_diff>0 → 右转，<0 → 左转，避免大圈
            turn_right = heading_diff > 0
            if abs(heading_diff) < 45:
                hdg_cmd = 10 if turn_right else 6  # 小幅转向
            else:
                hdg_cmd = 11 if turn_right else 5  # 中幅转向
            try:
                from utils.trace_logger import trace_throttle
                direction = '右转' if turn_right else '左转'
                trace_throttle(
                    key=f"formation:turning:{agent_id}",
                    min_steps=120,
                    标题="编队重整-转向",
                    env=env,
                    模块="advanced_formation_reset_manager",
                    类型="ACTION",
                    状态="PERIODIC",
                    我机=str(agent_id),
                    阶段="FORMATION_RESET",
                    战术="FORMATION_RESET",
                    说明="最短路径转向0°（避免大圈）",
                    数据={
                        "dir": direction,
                        "heading_deg": float(current_heading),
                        "target_heading_deg": 0.0,
                        "diff_deg": float(heading_diff),
                        "hdg_cmd": int(hdg_cmd),
                    },
                )
            except Exception:
                pass
        
        # 🔥 问题5修复：分阶段执行 + 重新校验保持阶段，防止误判后持续“保持”
        # 阶段1：编队重整到0度
        all_turned = True
        heading_diffs: Dict[str, float] = {}
        active_agents = self.active_agents or ['A0100', 'A0200']
        for aid in active_agents:
            if aid in env.agents and env.agents[aid].is_alive:
                aid_heading = np.rad2deg(env.agents[aid].get_property_value(c.attitude_psi_rad))
                aid_heading_diff = abs(TacticalUtils.normalize_angle_diff(0.0 - aid_heading))
                heading_diffs[aid] = aid_heading_diff
                if aid_heading_diff > self.heading_tolerance:
                    all_turned = False

        # 保持阶段安全校验：若任何一机偏离容差，回退到turning重新纠正
        if self.phase == 'holding_zero' and not all_turned:
            self.phase = 'turning'
            try:
                from utils.trace_logger import trace_throttle
                trace_throttle(
                    key="formation:holding_back_to_turning",
                    min_steps=60,
                    标题="编队重整-回退",
                    env=env,
                    模块="advanced_formation_reset_manager",
                    类型="STATE",
                    状态="CHANGED",
                    阶段="FORMATION_RESET",
                    战术="FORMATION_RESET",
                    说明="保持阶段检测到任一机超出容差→回退turning重新纠正",
                    数据={"diffs": {k: float(v) for k, v in heading_diffs.items()}, "tolerance_deg": float(self.heading_tolerance)},
                )
            except Exception:
                pass
        
        if all_turned:
            # 阶段1完成：所有飞机都到达0度
            if self.phase == 'turning':
                # 进入阶段2：保持0度航向
                self.phase = 'holding_zero'
                self.zero_heading_hold_start = env.current_step * env.time_interval
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="编队重整-阶段切换",
                        env=env,
                        模块="advanced_formation_reset_manager",
                        类型="STATE",
                        状态="CHANGED",
                        阶段="FORMATION_RESET",
                        战术="FORMATION_RESET",
                        说明="阶段1完成：双机航向均进入容差→进入保持阶段",
                        数据={"phase": "holding_zero", "diffs": {k: float(v) for k, v in heading_diffs.items()}, "tolerance_deg": float(self.heading_tolerance)},
                    )
                except Exception:
                    pass
            elif self.phase == 'holding_zero':
                # 阶段2：保持0度航向
                current_time = env.current_step * env.time_interval
                if self.zero_heading_hold_start is None:
                    self.zero_heading_hold_start = current_time
                
                hold_duration = current_time - self.zero_heading_hold_start
                if hold_duration >= self.zero_heading_hold_duration:
                    # 阶段2完成：进入阶段3（动态调整朝向目标）
                    self.phase = 'complete'
                    self.active = False
                    try:
                        from utils.trace_logger import trace_event
                        dur_total = float(current_time - float(self.start_time)) if self.start_time is not None else None
                        trace_event(
                            事件="编队重整-完成",
                            env=env,
                            模块="advanced_formation_reset_manager",
                            类型="SPAN",
                            状态="END",
                            阶段="FORMATION_RESET",
                            战术="FORMATION_RESET",
                            说明="保持阶段收敛完成：退出编队重整（后续由二次进攻/动态航向接管）",
                            数据={"hold_s": float(hold_duration), "total_s": dur_total},
                        )
                    except Exception:
                        pass
                else:
                    # 继续保持0度航向
                    try:
                        from utils.trace_logger import trace_throttle
                        trace_throttle(
                            key="formation:hold_progress",
                            min_steps=150,
                            标题="编队重整-保持进度",
                            env=env,
                            模块="advanced_formation_reset_manager",
                            类型="SPAN",
                            状态="PERIODIC",
                            阶段="FORMATION_RESET",
                            战术="FORMATION_RESET",
                            说明="保持阶段进行中",
                            数据={"hold_s": float(hold_duration), "hold_target_s": float(self.zero_heading_hold_duration)},
                        )
                    except Exception:
                        pass
        
        # 高度保持
        target_alt = 8000.0
        alt_diff = target_alt - current_alt
        if abs(alt_diff) < 200:
            alt_cmd = 7
        elif alt_diff > 0:
            alt_cmd = 9  # 小幅爬升
        else:
            alt_cmd = 5  # 小幅下降
        
        return alt_cmd, hdg_cmd, 3

    def is_formation_reset_complete(self, env, agents: List[str]) -> bool:
        """检查编队重置是否完成 - 分阶段检查"""
        # 🔥 如果已经在execute_formation_reset中标记完成，直接返回True
        if self.phase == 'complete':
            return True
            
        if not self.active:
            return False
            
        from envs.JSBSim.core.catalog import Catalog as c
        from tactical_utils import TacticalUtils
        
        current_time = env.current_step * env.time_interval
            
        # 🔥 问题5修复：使用严格的5度容差
        all_headings_good = True
        
        for aid in agents:
            if aid not in env.agents or not env.agents[aid].is_alive:
                continue
            
            aircraft = env.agents[aid]
            heading = np.rad2deg(aircraft.get_property_value(c.attitude_psi_rad))
            
            # 🔥 使用严格的5度容差
            heading_diff = abs(TacticalUtils.normalize_angle_diff(heading - self.target_heading))
            if heading_diff > self.heading_tolerance:  # 5度容差
                all_headings_good = False
                break
        
        if all_headings_good:
            # 如果还在turning阶段，进入holding_zero阶段
            if self.phase == 'turning':
                self.phase = 'holding_zero'
                self.zero_heading_hold_start = current_time
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="编队重整-阶段切换",
                        env=env,
                        模块="advanced_formation_reset_manager",
                        类型="STATE",
                        状态="CHANGED",
                        阶段="FORMATION_RESET",
                        战术="FORMATION_RESET",
                        说明="航向已对准→进入保持阶段(兜底检查路径)",
                        数据={"phase": "holding_zero", "tolerance_deg": float(self.heading_tolerance)},
                    )
                except Exception:
                    pass
                return False  # 还未完成，需要保持阶段
            elif self.phase == 'holding_zero':
                # 检查保持时间
                if self.zero_heading_hold_start is None:
                    self.zero_heading_hold_start = current_time
                
                hold_duration = current_time - self.zero_heading_hold_start
                if hold_duration >= self.zero_heading_hold_duration:
                    try:
                        from utils.trace_logger import trace_event
                        dur_total = float(current_time - float(self.start_time)) if self.start_time is not None else None
                        trace_event(
                            事件="编队重整-完成",
                            env=env,
                            模块="advanced_formation_reset_manager",
                            类型="SPAN",
                            状态="END",
                            阶段="FORMATION_RESET",
                            战术="FORMATION_RESET",
                            说明="保持阶段完成→重置完成(兜底检查路径)",
                            数据={"hold_s": float(hold_duration), "total_s": dur_total},
                        )
                    except Exception:
                        pass
                    self.active = False
                    self.phase = 'complete'
                    return True
                else:
                    return False  # 还在保持阶段
            else:
                try:
                    from utils.trace_logger import trace_event
                    dur_total = float(current_time - float(self.start_time)) if self.start_time is not None else None
                    trace_event(
                        事件="编队重整-完成",
                        env=env,
                        模块="advanced_formation_reset_manager",
                        类型="SPAN",
                        状态="END",
                        阶段="FORMATION_RESET",
                        战术="FORMATION_RESET",
                        说明="航向已对准→直接完成(兜底路径)",
                        数据={"total_s": dur_total},
                    )
                except Exception:
                    pass
                self.active = False
                self.phase = 'complete'
                return True
            
        return False
