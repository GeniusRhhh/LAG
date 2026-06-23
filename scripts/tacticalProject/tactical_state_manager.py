"""
战术状态管理模块
管理飞机阶段、机动状态、导弹状态等信息
从tactical_task.py中提取，提高代码可维护性

*** 重要变更 ***
实现单例模式，确保全局状态一致性，避免多模块修改冲突
"""
import logging
from tactical_types import TacticalPhase
from typing import Dict


class TacticalStateManager:
    """战术状态管理器 - 单例模式，管理所有状态信息"""
    
    def __init__(self):
        """初始化状态管理器（每个战术任务有自己独立的实例）"""
        
        # 阶段管理
        self.current_phase = TacticalPhase.BEYOND_NLT
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
        
        logging.info("[状态管理] 战术状态管理器(单例)初始化完成")
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例实例（仅用于测试或重新初始化）"""
        cls._instance = None
        cls._initialized = False
        logging.info("[状态管理] 战术状态管理器单例已重置")
    
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
            # 🔥 只为我方（A开头）输出阶段切换日志，敌方静默
            if agent_id.startswith('A'):
                logging.info(f"[阶段切换] {agent_id}: {old_phase} → {phase.value}")

            # ✅ 中文文件日志：阶段切换（不影响控制台输出）
            # 🔥 同样只记录我方
            if agent_id.startswith('A'):
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="阶段切换",
                        env=getattr(self, 'env', None),
                        我机=agent_id,
                        阶段=phase.value,
                        说明="状态管理器记录阶段变更",
                        数据={
                            "旧阶段": getattr(old_phase, 'value', str(old_phase)) if old_phase is not None else None,
                            "新阶段": phase.value,
                            "进入时间_s": float(current_time),
                        },
                    )
                except Exception:
                    pass
    
    def get_agent_phase(self, agent_id: str) -> TacticalPhase:
        """获取飞机当前阶段"""
        return self.agent_phases.get(agent_id, TacticalPhase.BEYOND_NLT)
    
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
        self.agent_phases[agent_id] = TacticalPhase.BEYOND_NLT
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
        
        logging.info(f"[MISSILE] {agent_id} fired -> target:{target_id}, remaining:{self.missiles_remaining[agent_id]}")
    
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
        self.current_phase = TacticalPhase.BEYOND_NLT
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
        self.missiles_fired.clear()
        self.lateral_split_active.clear()
        self.vertical_split_active.clear()
        self.lateral_split_targets.clear()
        self.vertical_split_targets.clear()
        self.lr_maneuver.clear()
        self.agent_targets.clear()
        self._return_log_counter.clear()
        logging.info("[状态重置] 组织所有状态已重置")
