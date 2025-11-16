"""
战术状态管理模块
管理飞机阶段、机动状态、导弹状态等信息
从tactical_task.py中提取，提高代码可维护性
"""
import logging
from tactical_types import TacticalPhase
from typing import Dict


class TacticalStateManager:
    """战术状态管理器 - 管理所有状态信息"""
    
    def __init__(self):
        """初始化状态管理器"""
        # 阶段管理
        self.current_phase = TacticalPhase.NLT_MELD
        self.agent_phases = {}  # 每架飞机的独立阶段
        self.phase_entry_time = {}  # 阶段进入时间
        
        # 机动状态
        self.maneuver_states = {}  # 复杂机动状态（tactical_crank, notch_back等）
        self.short_skate_states = {}  # Short Skate机动状态
        self.short_skate_start_time = {}  # Short Skate开始时间
        self.beam_maneuver_state = {}  # Beam机动状态
        self.formation_state = {}  # 队形状态
        
        # 导弹状态
        self.missile_launched = {}  # 是否发射导弹
        self.last_missile_launch_time = {}  # 最后发射时间
        self.missiles_remaining = {}  # 剩余导弹数
        self.missiles_fired = {}  # 已发射导弹数
        self.missile_cooldown = 10.0  # 导弹发射冷却时间（10秒）
        
        # 分离状态
        self.lateral_split_active = {}  # 横向分离状态
        self.vertical_split_active = {}  # 垂直分离状态
        self.lateral_split_targets = {}  # 横向分离目标
        self.vertical_split_targets = {}  # 垂直分离目标
        
        # 决策结果缓存
        self.lr_maneuver = {}  # LR阶段机动决策
        self.evasion_maneuver = 'SHORT_SKATE'  # 规避机动类型
        
        # 目标跟踪
        self.agent_targets = {}  # 每架飞机的目标
        
        # 日志计数器
        self._return_log_counter = {}  # 返航日志计数
    
    def set_agent_phase(self, agent_id: str, phase: TacticalPhase, current_time: float):
        """
        设置飞机的战术阶段
        
        Args:
            agent_id: 飞机ID
            phase: 战术阶段
            current_time: 当前时间
        """
        old_phase = self.agent_phases.get(agent_id)
        if old_phase != phase:
            self.agent_phases[agent_id] = phase
            self.phase_entry_time[agent_id] = current_time
            logging.info(f"📍 [{agent_id}] 阶段切换: {old_phase} → {phase.value}")
    
    def get_agent_phase(self, agent_id: str) -> TacticalPhase:
        """获取飞机当前阶段"""
        return self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
    
    def get_phase_duration(self, agent_id: str, current_time: float) -> float:
        """获取当前阶段持续时间"""
        entry_time = self.phase_entry_time.get(agent_id, 0)
        return current_time - entry_time
    
    def init_agent_states(self, agent_id: str, max_missiles: int = 4):
        """
        初始化飞机状态
        
        Args:
            agent_id: 飞机ID
            max_missiles: 最大导弹数量
        """
        self.agent_phases[agent_id] = TacticalPhase.NLT_MELD
        self.phase_entry_time[agent_id] = 0
        self.missile_launched[agent_id] = False
        self.last_missile_launch_time[agent_id] = -999
        self.missiles_remaining[agent_id] = max_missiles
        self.missiles_fired[agent_id] = 0
        self.agent_targets[agent_id] = None
    
    def record_missile_launch(self, agent_id: str, current_time: float, target_id: str = None):
        """
        记录导弹发射
        
        Args:
            agent_id: 发射飞机ID
            current_time: 发射时间
            target_id: 目标ID（可选）
        """
        self.missile_launched[agent_id] = True
        self.last_missile_launch_time[agent_id] = current_time
        remaining = self.missiles_remaining.get(agent_id, 0)
        if remaining > 0:
            self.missiles_remaining[agent_id] = remaining - 1
        
        logging.info(f"🚀 [{agent_id}] 发射导弹 → 目标:{target_id}, 剩余:{self.missiles_remaining[agent_id]}")
    
    def can_launch_missile(self, agent_id: str, current_time: float, cooldown: float = 5.0) -> bool:
        """
        检查是否可以发射导弹
        
        Args:
            agent_id: 飞机ID
            current_time: 当前时间
            cooldown: 冷却时间（秒）
            
        Returns:
            True表示可以发射
        """
        # 检查剩余导弹
        remaining = self.missiles_remaining.get(agent_id, 0)
        if remaining <= 0:
            return False
        
        # 检查冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if last_launch > 0 and (current_time - last_launch) < cooldown:
            return False
        
        return True
    
    def is_in_guidance_protection(self, agent_id: str, current_time: float, protection_time: float = 15.0) -> bool:
        """
        检查是否处于导弹制导保护期
        
        Args:
            agent_id: 飞机ID
            current_time: 当前时间
            protection_time: 保护时间（秒）
            
        Returns:
            True表示处于保护期
        """
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if last_launch > 0 and (current_time - last_launch) < protection_time:
            return True
        return False
    
    def clear_maneuver_state(self, agent_id: str, maneuver_type: str = None):
        """
        清除机动状态
        
        Args:
            agent_id: 飞机ID
            maneuver_type: 机动类型（None表示清除所有）
        """
        if maneuver_type is None:
            # 清除所有机动状态
            if agent_id in self.maneuver_states:
                del self.maneuver_states[agent_id]
            if agent_id in self.short_skate_states:
                del self.short_skate_states[agent_id]
            if agent_id in self.beam_maneuver_state:
                del self.beam_maneuver_state[agent_id]
        else:
            # 清除特定机动状态
            if maneuver_type == 'short_skate' and agent_id in self.short_skate_states:
                del self.short_skate_states[agent_id]
            elif maneuver_type == 'beam' and agent_id in self.beam_maneuver_state:
                del self.beam_maneuver_state[agent_id]
            elif agent_id in self.maneuver_states and self.maneuver_states[agent_id].get('type') == maneuver_type:
                del self.maneuver_states[agent_id]
    
    def set_target(self, agent_id: str, target_id: str):
        """设置飞机目标"""
        self.agent_targets[agent_id] = target_id
    
    def get_target(self, agent_id: str) -> str:
        """获取飞机目标"""
        return self.agent_targets.get(agent_id)
    
    def reset_all(self):
        """重置所有状态"""
        self.current_phase = TacticalPhase.NLT_MELD
        self.agent_phases.clear()
        self.phase_entry_time.clear()
        self.maneuver_states.clear()
        self.short_skate_states.clear()
        self.short_skate_start_time.clear()
        self.beam_maneuver_state.clear()
        self.formation_state.clear()
        self.missile_launched.clear()
        self.last_missile_launch_time.clear()
        self.missiles_remaining.clear()
        self.lateral_split_active.clear()
        self.vertical_split_active.clear()
        self.lateral_split_targets.clear()
        self.vertical_split_targets.clear()
        self.lr_maneuver.clear()
        self.agent_targets.clear()
        self._return_log_counter.clear()
    
    def get_status_summary(self, agent_id: str) -> Dict:
        """
        获取飞机状态摘要
        
        Args:
            agent_id: 飞机ID
            
        Returns:
            状态摘要字典
        """
        return {
            'phase': self.get_agent_phase(agent_id).value,
            'missiles_remaining': self.missiles_remaining.get(agent_id, 0),
            'last_launch_time': self.last_missile_launch_time.get(agent_id, -999),
            'target': self.get_target(agent_id),
            'in_short_skate': agent_id in self.short_skate_states,
            'in_maneuver': agent_id in self.maneuver_states,
            'maneuver_type': self.maneuver_states.get(agent_id, {}).get('type', None)
        }
