#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
节点决策器模块（Node Decision Maker Module）

功能：
    1. 管理各控制距离节点的战术决策
    2. 根据态势和威胁评估选择合适的战术和机动
    3. 处理两轮攻击的决策逻辑
    4. 支持动态战术切换

作者：TacticalProject重构组
版本：2.0.0
日期：2024-11-13
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from enum import Enum
from dataclasses import dataclass, field
from .node_decision_methods import NodeDecisionMethods


class ControlDistance(Enum):
    """控制距离节点枚举"""
    NLT = "NLT"      # 120km
    MELD = "MELD"    # 100km
    MTR = "MTR"      # 80km
    LR = "LR"        # 78km
    TR = "TR"        # 75km
    DOR = "DOR"      # 70km
    DR = "DR"        # 65km
    MAR = "MAR"      # 40km
    # 第二轮攻击节点
    MTR2 = "MTR2"    # 80km (第二轮)
    LR2 = "LR2"      # 78km (第二轮)
    TR2 = "TR2"      # 75km (第二轮)


# ==================== 数据类定义 ====================

@dataclass
class DecisionResult:
    """
    决策结果数据类
    封装节点决策的输出
    """
    node: str                              # 决策节点
    selected_tactic: str                   # 选择的战术
    selected_maneuver: str                 # 选择的机动
    formation_adjustment: Optional[Dict]   # 编队调整参数
    reengage: bool = False                # 是否重新进攻
    retreat: bool = False                 # 是否撤退
    custom_params: Dict = field(default_factory=dict)  # 自定义参数


@dataclass
class NodeContext:
    """
    节点决策上下文
    包含决策所需的所有信息
    """
    env: Any                              # 环境对象
    agent_id: str                         # 智能体ID
    current_time: float                   # 当前时间
    distance: float                       # 到敌机距离
    threat_level: float                   # 威胁等级
    situation: str                        # 态势评估
    enemy_intent: str                     # 敌方意图
    our_intent: str                       # 我方意图
    my_aircraft: list                     # 我方飞机列表
    enemy_aircraft: list                  # 敌方飞机列表
    is_second_attack: bool = False       # 是否第二轮攻击
    custom_data: Dict = field(default_factory=dict)  # 自定义数据


# ==================== 主类定义 ====================

class NodeDecisionMaker(NodeDecisionMethods):
    """
    节点决策器（Node Decision Maker）
    
    职责：
        1. 在各控制距离节点进行战术决策
        2. 管理决策历史和状态
        3. 支持两轮攻击决策
        4. 处理战术切换条件
    
    设计原则：
        - 基于项目说明文档的决策框架
        - 遵循四级控制距离的决策权限
        - 支持动态战术调整
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化节点决策器
        
        Args:
            logger: 日志记录器（可选）
        """
        # 🔥 先调用父类初始化，确保团队决策状态正确设置
        super().__init__()
        
        # 日志配置
        self.logger = logger or logging.getLogger(__name__)
        
        # 决策状态管理
        self.node_decisions: Dict[str, DecisionResult] = {}  # 各节点的决策记录
        self.decision_history: List[DecisionResult] = []     # 决策历史
        
        # 🔥 修复1: 飞机级别的DR状态管理，避免共享状态冲突
        self.agent_dr_windows: Dict[str, float] = {}        # agent_id -> DR窗口开始时间
        self.agent_dr_decisions: Dict[str, Dict] = {}       # agent_id -> DR决策状态
        self.dr_window_start: Optional[float] = None        # 保留兼容性（已弃用）
        
        # 决策参数配置
        self.decision_params = self._init_decision_params()
        
        # 决策表接口（需要外部注入）
        self.decision_table = None
        self.tactical_selector = None
        self.threat_evaluator = None
        self.situation_evaluator = None
        self.intent_recognizer = None
        
        self.logger.info("✅ [NodeDecisionMaker] 节点决策器初始化完成")
    
    def _init_decision_params(self) -> Dict:
        """
        初始化决策参数配置
        
        Returns:
            决策参数字典
        """
        params = {
            'NLT': {
                'level': 'STRATEGY',      # 一级节点
                'allow_tactic_change': True,         # 允许战术切换
                'allow_formation_adjust': True,      # 允许编队调整
                'decision_priority': 1,              # 决策优先级
            },
            'MELD': {
                'level': 'TACTIC',       # 二级节点
                'allow_tactic_change': True,
                'allow_formation_adjust': True,
                'decision_priority': 2,
            },
            'MTR': {
                'level': 'MANEUVER',     # 三级节点
                'allow_tactic_change': False,        # 不允许战术切换
                'allow_maneuver_select': True,       # 允许机动选择
                'decision_priority': 3,
            },
            'LR': {
                'level': 'PARAMETER',    # 四级节点
                'allow_tactic_change': False,
                'allow_parameter_adjust': True,      # 允许参数调整
                'decision_priority': 4,
            },
            'DOR': {
                'level': 'TACTIC',       # 二级节点
                'allow_tactic_change': True,
                'allow_defensive_switch': True,      # 允许切换到防御
                'decision_priority': 2,
            },
            'DR': {
                'level': 'TACTIC',       # 二级节点
                'allow_reengage': True,              # 允许重新进攻
                'time_window': 10.0,                 # DR时间窗口（秒）
                'decision_priority': 2,
            },
            'MAR': {
                'level': 'MANEUVER',     # 三级节点
                'force_defensive': True,             # 强制防御
                'decision_priority': 3,
            },

            # 第二轮攻击节点（与 decide_at_mtr2/lr2/tr2 对齐）
            'MTR2': {
                'level': 'MANEUVER',
                'allow_tactic_change': False,
                'allow_maneuver_select': True,
                'decision_priority': 3,
            },
            'LR2': {
                'level': 'PARAMETER',
                'allow_tactic_change': False,
                'allow_parameter_adjust': True,
                'decision_priority': 4,
            },
            'TR2': {
                'level': 'MANEUVER',
                'allow_tactic_change': False,
                'allow_maneuver_select': True,
                'decision_priority': 3,
            },
        }

        return params
