"""
编队管理模块
管理2×双机编队的冷热循环
"""
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


class PatrolPhase(Enum):
    """巡逻阶段"""
    HOT = "HOT"    # 热段（朝向靶眼）
    COLD = "COLD"  # 冷段（背离靶眼）


class FormationRole(Enum):
    """编队角色"""
    LEAD = "LEAD"        # 长机
    WINGMAN = "WINGMAN"  # 僚机


@dataclass
class FormationMember:
    """编队成员"""
    agent_id: str
    role: FormationRole
    initial_phase: PatrolPhase
    x_rel: float  # 战场相对X坐标
    y_rel: float  # 战场相对Y坐标


class FormationManager:
    """编队管理器"""
    
    def __init__(self, wingman_y_offset: float = 100.0,
                 lead_x: Tuple[float, float] = (75, 125),
                 wingman_x: Tuple[float, float] = (25, 175)):
        """
        Args:
            wingman_y_offset: 僚机Y偏移
            lead_x: (左长机X, 右长机X)
            wingman_x: (左僚机X, 右僚机X)
        """
        self.wingman_y = wingman_y_offset
        
        # 编队定义
        self.formations: Dict[int, List[FormationMember]] = {
            0: [  # 左编队
                FormationMember('A0100', FormationRole.LEAD, PatrolPhase.HOT, lead_x[0], 0),
                FormationMember('A0200', FormationRole.WINGMAN, PatrolPhase.COLD, wingman_x[0], wingman_y_offset),
            ],
            1: [  # 右编队
                FormationMember('A0300', FormationRole.LEAD, PatrolPhase.HOT, lead_x[1], 0),
                FormationMember('A0400', FormationRole.WINGMAN, PatrolPhase.COLD, wingman_x[1], wingman_y_offset),
            ],
        }
        
        # 当前阶段状态
        self.current_phase: Dict[str, PatrolPhase] = {}
        for fid, members in self.formations.items():
            for m in members:
                self.current_phase[m.agent_id] = m.initial_phase
        
        # 循环计数
        self.cycle_count = 0
    
    def get_formation_id(self, agent_id: str) -> int:
        """获取飞机所属编队ID"""
        for fid, members in self.formations.items():
            if any(m.agent_id == agent_id for m in members):
                return fid
        return -1
    
    def get_role(self, agent_id: str) -> Optional[FormationRole]:
        """获取飞机角色"""
        for members in self.formations.values():
            for m in members:
                if m.agent_id == agent_id:
                    return m.role
        return None
    
    def is_lead(self, agent_id: str) -> bool:
        """是否为长机"""
        return self.get_role(agent_id) == FormationRole.LEAD
    
    def get_wingman(self, lead_id: str) -> Optional[str]:
        """获取长机对应的僚机"""
        fid = self.get_formation_id(lead_id)
        if fid < 0:
            return None
        for m in self.formations[fid]:
            if m.role == FormationRole.WINGMAN:
                return m.agent_id
        return None
    
    def get_current_phase(self, agent_id: str) -> PatrolPhase:
        """获取当前巡逻阶段"""
        return self.current_phase.get(agent_id, PatrolPhase.HOT)
    
    def check_cycle_switch(self, positions: Dict[str, Tuple[float, float]], tolerance: float = 2.0) -> bool:
        """检查是否需要冷热交替
        
        条件：长机到达僚机初始Y位置，僚机到达长机初始Y位置
        """
        for fid, members in self.formations.items():
            lead = next(m for m in members if m.role == FormationRole.LEAD)
            wingman = next(m for m in members if m.role == FormationRole.WINGMAN)
            
            lead_pos = positions.get(lead.agent_id)
            wingman_pos = positions.get(wingman.agent_id)
            
            if not lead_pos or not wingman_pos:
                continue
            
            lead_y = lead_pos[1]
            wingman_y = wingman_pos[1]
            
            # 检查是否到达交替位置
            if self.current_phase[lead.agent_id] == PatrolPhase.HOT:
                # 长机热段朝北，应该到达wingman_y
                if lead_y >= self.wingman_y - tolerance and wingman_y <= tolerance:
                    return True
            else:
                # 长机冷段朝南，应该到达0
                if lead_y <= tolerance and wingman_y >= self.wingman_y - tolerance:
                    return True
        
        return False
    
    def execute_cycle_switch(self):
        """执行冷热交替"""
        for agent_id in self.current_phase:
            if self.current_phase[agent_id] == PatrolPhase.HOT:
                self.current_phase[agent_id] = PatrolPhase.COLD
            else:
                self.current_phase[agent_id] = PatrolPhase.HOT
        self.cycle_count += 1
