"""
决策制定系统
实现4级决策：策略决策 → 战术决策 → 机动决策 → 参数决策
"""

import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from enum import Enum
from dataclasses import dataclass

from .intent_recognizer import EnemyIntent, FriendlyIntent
from .situation_evaluator import SituationScore, TacticalPhase


class ThreatResponse(Enum):
    """威胁响应类型"""
    CONTINUE = "CONTINUE"      # 继续当前任务
    EVADE = "EVADE"           # 执行规避
    RETREAT = "RETREAT"       # 撤退


@dataclass
class ThreatDecision:
    """威胁决策结果"""
    response: ThreatResponse
    reason: str
    threat_level: float


class DecisionMaker:
    """
    决策制定器
    根据态势、威胁、意图进行多层级决策
    """
    
    def __init__(self):
        """初始化决策制定器"""
        # 威胁阈值
        self.retreat_threat_threshold = 0.8    # 撤退威胁阈值
        self.evade_threat_threshold = 0.8      # 规避威胁阈值（单项）
        self.evade_multi_threshold = 2         # 需要多少个单项超阈值才规避
        
        logging.info("✅ 决策制定系统初始化完成")
    
    def make_threat_decision(
        self,
        my_intent: FriendlyIntent,
        enemy_intent: EnemyIntent,
        threat_score: float,
        situation: SituationScore,
        phase: TacticalPhase
    ) -> ThreatDecision:
        """
        制定威胁响应决策
        
        Args:
            my_intent: 我方意图
            enemy_intent: 敌方意图
            threat_score: 总威胁值
            situation: 态势得分
            phase: 当前阶段
            
        Returns:
            ThreatDecision: 威胁决策结果
        """
        # 激进肃清：永不撤退，继续进攻
        if my_intent == FriendlyIntent.AGGRESSIVE_CLEAR:
            return ThreatDecision(
                response=ThreatResponse.CONTINUE,
                reason="激进肃清意图，不顾威胁继续进攻",
                threat_level=threat_score
            )
        
        # 防御意图：特殊逻辑
        if my_intent == FriendlyIntent.DEFENSIVE:
            return self._make_defensive_decision(enemy_intent, threat_score, phase)
        
        # 保守肃清：根据威胁程度决策
        if my_intent == FriendlyIntent.CONSERVATIVE_CLEAR:
            return self._make_conservative_decision(
                enemy_intent,
                threat_score,
                situation,
                phase
            )
        
        # 默认：继续
        return ThreatDecision(
            response=ThreatResponse.CONTINUE,
            reason="默认继续当前任务",
            threat_level=threat_score
        )
    
    def _make_defensive_decision(
        self,
        enemy_intent: EnemyIntent,
        threat_score: float,
        phase: TacticalPhase
    ) -> ThreatDecision:
        """
        防御意图决策逻辑
        
        目标：逼迫敌机放弃进攻意图
        """
        # 敌机逃逸：任务完成，可以脱离
        if enemy_intent == EnemyIntent.ESCAPE:
            return ThreatDecision(
                response=ThreatResponse.RETREAT,
                reason="敌机逃逸，防御任务完成",
                threat_level=threat_score
            )
        
        # 敌机中立且在TR阶段后：第一轮攻击结束，可以脱离
        if enemy_intent in [EnemyIntent.JAMMING, EnemyIntent.RECONNAISSANCE, EnemyIntent.DEFENSIVE]:
            if phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR, TacticalPhase.DR_MAR, TacticalPhase.BEYOND_MAR]:
                return ThreatDecision(
                    response=ThreatResponse.RETREAT,
                    reason="敌机中立意图且第一轮攻击结束",
                    threat_level=threat_score
                )
        
        # 其他情况：继续防御
        return ThreatDecision(
            response=ThreatResponse.CONTINUE,
            reason="继续防御态势",
            threat_level=threat_score
        )
    
    def _make_conservative_decision(
        self,
        enemy_intent: EnemyIntent,
        threat_score: float,
        situation: SituationScore,
        phase: TacticalPhase
    ) -> ThreatDecision:
        """
        保守肃清决策逻辑
        
        检查顺序：撤退 → 规避 → 继续
        """
        # 1. 检查是否需要撤退
        # 条件：敌方进攻意图 + 总威胁度>0.8
        if enemy_intent in [EnemyIntent.ATTACK, EnemyIntent.COORDINATED, EnemyIntent.FEINT]:
            if threat_score > self.retreat_threat_threshold:
                return ThreatDecision(
                    response=ThreatResponse.RETREAT,
                    reason=f"敌方进攻且总威胁度{threat_score:.2f}>0.8，执行撤退",
                    threat_level=threat_score
                )
            
            # 或者：2个以上单项威胁度>0.8
            high_threat_count = sum([
                1 if 1 - situation.angle > self.evade_threat_threshold else 0,
                1 if 1 - situation.distance > self.evade_threat_threshold else 0,
                1 if 1 - situation.altitude > self.evade_threat_threshold else 0,
                1 if 1 - situation.speed > self.evade_threat_threshold else 0,
                1 if 1 - situation.detection > self.evade_threat_threshold else 0
            ])
            
            if high_threat_count >= 2:
                return ThreatDecision(
                    response=ThreatResponse.RETREAT,
                    reason=f"敌方进攻且{high_threat_count}个单项威胁度>0.8，执行撤退",
                    threat_level=threat_score
                )
        
        # 2. 检查是否需要规避
        # 条件：敌方进攻或中立 + 1-2个单项威胁度>0.8
        if enemy_intent in [EnemyIntent.ATTACK, EnemyIntent.COORDINATED, EnemyIntent.FEINT,
                           EnemyIntent.JAMMING, EnemyIntent.RECONNAISSANCE, EnemyIntent.DEFENSIVE]:
            high_threat_items = []
            if 1 - situation.angle > self.evade_threat_threshold:
                high_threat_items.append("角度")
            if 1 - situation.distance > self.evade_threat_threshold:
                high_threat_items.append("距离")
            if 1 - situation.altitude > self.evade_threat_threshold:
                high_threat_items.append("高度")
            if 1 - situation.speed > self.evade_threat_threshold:
                high_threat_items.append("速度")
            if 1 - situation.detection > self.evade_threat_threshold:
                high_threat_items.append("探测")
            
            if 1 <= len(high_threat_items) <= 2:
                return ThreatDecision(
                    response=ThreatResponse.EVADE,
                    reason=f"单项威胁过高({'、'.join(high_threat_items)})，执行规避",
                    threat_level=threat_score
                )
        
        # 3. 继续当前任务
        return ThreatDecision(
            response=ThreatResponse.CONTINUE,
            reason="威胁可控，继续执行任务",
            threat_level=threat_score
        )
    
    def should_launch_missile(
        self,
        phase: TacticalPhase,
        distance: float,
        threat_decision: ThreatDecision,
        has_launched: bool
    ) -> bool:
        """
        决定是否发射导弹
        
        Args:
            phase: 当前阶段
            distance: 与目标距离（米）
            threat_decision: 威胁决策
            has_launched: 是否已发射
            
        Returns:
            bool: 是否应该发射导弹
        """
        # 已经发射过了
        if has_launched:
            return False
        
        # 威胁决策为撤退：不发射
        if threat_decision.response == ThreatResponse.RETREAT:
            return False
        
        # 必须在LR_TR阶段
        if phase != TacticalPhase.LR_TR:
            return False
        
        # 距离检查（70-85km）
        distance_km = distance / 1000.0
        if not (70 <= distance_km <= 85):
            return False
        
        return True
    
    def should_enter_second_round(
        self,
        phase: TacticalPhase,
        my_intent: FriendlyIntent,
        enemy_intent: EnemyIntent,
        threat_score: float,
        enemy_alive: bool
    ) -> bool:
        """
        决定是否发起第二轮进攻
        
        在DR节点做出决策
        """
        # 敌机已被击落：不需要第二轮
        if not enemy_alive:
            return False
        
        # 必须在DR阶段或之后
        if phase not in [TacticalPhase.DOR_DR, TacticalPhase.DR_MAR]:
            return False
        
        # 激进肃清：必定二次进攻
        if my_intent == FriendlyIntent.AGGRESSIVE_CLEAR:
            return True
        
        # 防御意图：不进行二次进攻
        if my_intent == FriendlyIntent.DEFENSIVE:
            return False
        
        # 保守肃清：根据态势判断
        if my_intent == FriendlyIntent.CONSERVATIVE_CLEAR:
            # 敌机逃逸：不追击
            if enemy_intent == EnemyIntent.ESCAPE:
                return False
            
            # 威胁过高：不二次进攻
            if threat_score > 0.65:
                return False
            
            # 其他情况：可以二次进攻
            return True
        
        return False
    
    def select_tactical_maneuver(
        self,
        phase: TacticalPhase,
        threat_decision: ThreatDecision,
        distance: float
    ) -> str:
        """
        选择机动动作
        
        返回机动动作名称
        """
        # 根据威胁决策选择
        if threat_decision.response == ThreatResponse.RETREAT:
            # 撤退：Short Skate或Notch Back
            if phase in [TacticalPhase.DOR_DR, TacticalPhase.DR_MAR]:
                return "SHORT_SKATE"
            else:
                return "NOTCH_BACK"
        
        elif threat_decision.response == ThreatResponse.EVADE:
            # 规避：Beam机动
            return "BEAM"
        
        else:
            # 继续任务：根据阶段选择
            if phase == TacticalPhase.LR_TR:
                return "CRANK"  # 中制导阶段保持Crank
            elif phase in [TacticalPhase.TR_DOR, TacticalPhase.DOR_DR]:
                return "SHORT_SKATE"  # 规避阶段
            else:
                return "LEVEL_FLIGHT"  # 默认平飞
