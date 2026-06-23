#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
节点决策方法模块（Node Decision Methods Module）

包含NodeDecisionMaker的具体决策方法实现

作者：TacticalProject重构组
版本：2.0.0
日期：2024-11-13
"""

from typing import Dict, Any, Optional, Tuple
import logging
import numpy as np
import os


class NodeDecisionMethods:
    """
    节点决策方法集合
    包含各控制距离节点的具体决策实现
    """
    
    def __init__(self):
        """初始化节点决策方法"""
        # 🔥 新增：团队统一决策状态管理
        self.team_decisions = {}  # 存储团队决策状态
        self.decision_consensus_cache = {}  # 决策共识缓存
        self.decision_table = None
        self.last_decision = None
    
    # ==================== 公共接口 ====================
    
    def make_decision(self, node: str, context: Any) -> Dict[str, Any]:
        """
        根据节点类型进行决策（主入口方法）
        
        Args:
            node: 节点名称
            context: 决策上下文
        
        Returns:
            决策结果字典
        """
        # 分发到具体节点的决策方法
        if node == 'NLT':
            return self.decide_at_nlt(context)
        elif node == 'MELD':
            return self.decide_at_meld(context)
        elif node == 'MTR':
            return self.decide_at_mtr(context)
        elif node == 'LR':
            return self.decide_at_lr(context)
        elif node == 'TR':
            return self.decide_at_tr(context)
        elif node == 'DOR':
            return self.decide_at_dor(context)
        elif node == 'DR':
            return self.decide_at_dr(context)
        elif node == 'MAR':
            return self.decide_at_mar(context)
        # 第二轮攻击节点
        elif node == 'MTR2':
            return self.decide_at_mtr2(context)
        elif node == 'LR2':
            return self.decide_at_lr2(context)
        elif node == 'TR2':
            return self.decide_at_tr2(context)
        else:
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="节点决策-未知节点",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="EXCEPTION",
                    状态="FAIL",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段=str(node),
                    说明="收到未知节点名，返回兜底直飞",
                    数据={"node": str(node)},
                )
            except Exception:
                pass
            return {'tactic': None, 'maneuver': 'straight'}
    
    # ==================== 一级节点决策（策略级） ====================
    
    def decide_at_nlt(self, context: Any) -> Dict[str, Any]:
        """
        NLT节点决策（120km）
        
        权限：
            - 可以选择战术
            - 可以调整编队
            - 可以设定意图
        
        Returns:
            决策结果
        """
        # ✅ 文件复盘：NLT节点决策输入（节流）
        try:
            from utils.trace_logger import trace_throttle

            trace_throttle(
                key=f"node:NLT:enter:{context.agent_id}",
                min_steps=80,
                标题="NLT节点-进入",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="NLT",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="一级节点：决策表候选 + 战术选择器输出",
                数据={
                    "distance_km": float(getattr(context, "distance", 0.0)) / 1000.0,
                    "our_intent": str(getattr(context, "our_intent", None)),
                    "enemy_intent": str(getattr(context, "enemy_intent", None)),
                    "situation": str(getattr(context, "situation", None)),
                },
            )
        except Exception:
            pass
        
        # 1. 从决策表查询候选战术
        if self.decision_table:
            candidates = self.decision_table.query_candidates(
                'NLT',
                context.our_intent,
                context.enemy_intent,
                context.situation
            )
        else:
            # 默认候选
            candidates = ['PINCER_ATTACK', 'DRAG_SHOOT', 'HIGH_LOW_ATTACK']
        
        # 2. 使用战术选择器选择最优战术
        if self.tactical_selector:
            selected_tactic = self.tactical_selector.select_tactic(
                'NLT',
                context.our_intent,
                context.enemy_intent,
                context.situation,
                context.my_aircraft,
                context.enemy_aircraft,
                context.env,
                context.custom_data.get('current_tactic')
            )
        else:
            # 默认选择第一个
            selected_tactic = candidates[0] if candidates else 'PINCER_ATTACK'
        
        # 3. 根据战术确定编队调整
        formation_adjustment = self._determine_formation_adjustment(selected_tactic)

        # ✅ 文件复盘：NLT节点决策输出（节流）
        try:
            from utils.trace_logger import trace_throttle

            trace_throttle(
                key=f"node:NLT:out:{context.agent_id}",
                min_steps=80,
                标题="NLT节点-输出",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="RETURN",
                我机=str(getattr(context, "agent_id", None)),
                阶段="NLT",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="输出战术/编队调整（用于解释为什么选此模板）",
                数据={
                    "candidates": list(candidates) if isinstance(candidates, (list, tuple)) else None,
                    "selected_tactic": str(selected_tactic),
                    "formation": formation_adjustment,
                },
            )
        except Exception:
            pass
        
        return {
            'node': 'NLT',
            'tactic': selected_tactic,
            'maneuver': 'straight',
            'formation': formation_adjustment,
            'allow_switch': True
        }
    
    # ==================== 二级节点决策（战术级） ====================
    
    def decide_at_meld(self, context: Any) -> Dict[str, Any]:
        """
        MELD节点决策（100km）
        
        权限：
            - 可以切换战术
            - 可以调整编队
            - 重点进行态势融合
        
        Returns:
            决策结果
        """
        # ✅ 文件复盘：MELD节点输入（节流）
        try:
            from utils.trace_logger import trace_throttle

            trace_throttle(
                key=f"node:MELD:enter:{context.agent_id}",
                min_steps=80,
                标题="MELD节点-进入",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="MELD",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="二级节点：根据切换条件决定是否重新选模板",
                数据={
                    "distance_km": float(getattr(context, "distance", 0.0)) / 1000.0,
                    "our_intent": str(getattr(context, "our_intent", None)),
                    "enemy_intent": str(getattr(context, "enemy_intent", None)),
                    "situation": str(getattr(context, "situation", None)),
                },
            )
        except Exception:
            pass
        
        # 1. 检查是否需要切换战术
        need_switch = self._check_tactic_switch_condition(context)
        
        if need_switch:
            # 重新选择战术
            if self.decision_table:
                candidates = self.decision_table.query_candidates(
                    'MELD',
                    context.our_intent,
                    context.enemy_intent,
                    context.situation
                )
            else:
                candidates = ['PINCER_ATTACK', 'HIGH_LOW_ATTACK']
            
            if self.tactical_selector and candidates:
                selected_tactic = self.tactical_selector.select_tactic(
                    'MELD',
                    context.our_intent,
                    context.enemy_intent,
                    context.situation,
                    context.my_aircraft,
                    context.enemy_aircraft,
                    context.env,
                    context.custom_data.get('current_tactic')
                )
            else:
                selected_tactic = None
        else:
            selected_tactic = None  # 保持当前战术
        
        # 2. 确定编队调整（根据战术执行MELD阶段的编队调整）
        current_tactic = context.custom_data.get('current_tactic', 'PINCER_ATTACK')
        formation_adjustment = self._get_meld_formation_adjustment(
            selected_tactic or current_tactic
        )
        
        # 文件-only：不再输出控制台日志

        # ✅ 文件复盘：MELD节点输出（节流）
        try:
            from utils.trace_logger import trace_throttle

            trace_throttle(
                key=f"node:MELD:out:{context.agent_id}",
                min_steps=80,
                标题="MELD节点-输出",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="RETURN",
                我机=str(getattr(context, "agent_id", None)),
                阶段="MELD",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="输出战术切换意图与编队调整",
                数据={
                    "need_switch": bool(need_switch),
                    "selected_tactic": str(selected_tactic) if selected_tactic is not None else None,
                    "formation": formation_adjustment,
                },
            )
        except Exception:
            pass
        
        return {
            'node': 'MELD',
            'tactic': selected_tactic,
            'maneuver': 'straight',
            'formation': formation_adjustment,
            'allow_switch': need_switch
        }
    
    def decide_at_dor(self, context: Any) -> Dict[str, Any]:
        """
        DOR节点决策（70km）
        
        权限：
            - 可以切换到防御战术
            - 重点评估威胁
        
        Returns:
            决策结果
        """
        # ✅ 文件复盘：DOR节点输入/输出（变化触发）
        try:
            from utils.trace_logger import trace_if_changed
            sig = (
                str(getattr(context, "our_intent", None)),
                str(getattr(context, "situation", None)),
                str(getattr(context, "enemy_intent", None)),
                float(getattr(context, "threat_level", 0.0)),
            )
            trace_if_changed(
                key=f"node:DOR:in:{getattr(context, 'agent_id', '')}",
                value=sig,
                标题="DOR节点-输入",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="DOR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="DOR：按意图×态势×敌意图确定允许集合(6/7)，再按威胁阈值稳定选择",
                数据={
                    "distance_km": float(getattr(context, "distance", 0.0)) / 1000.0,
                    "our_intent": str(getattr(context, "our_intent", None)),
                    "enemy_intent": str(getattr(context, "enemy_intent", None)),
                    "situation": str(getattr(context, "situation", None)),
                    "threat_level": float(getattr(context, "threat_level", 0.0)),
                },
            )
        except Exception:
            pass
        
        # ✅ 按用户给定规则：DOR节点输出战术规避(6)/战术回转(7)的集合（并利用threat_level在可选集合内二选一）
        our = context.our_intent
        sit = context.situation
        enemy = context.enemy_intent  # ATTACK/NEUTRAL/RETREAT(ESCAPE)

        # 先确定允许的战术集合（只影响6/7，不改变模板内部动作序列）
        allow = []
        if our == 'AGGRESSIVE_CLEAR':
            # 我方占优：只规避6；敌方占优：6或7
            allow = ['TACTICAL_EVASION'] if sit == 'ADVANTAGE' else ['TACTICAL_EVASION', 'TACTICAL_TURN']
        elif our == 'CONSERVATIVE_CLEAR':
            # 全态势：6或7
            allow = ['TACTICAL_EVASION', 'TACTICAL_TURN']
        else:  # DEFENSIVE
            # 全态势全意图：只回转7
            allow = ['TACTICAL_TURN']

        # 在允许集合中用威胁等级做一个稳定选择：威胁越高越倾向回转
        if allow == ['TACTICAL_TURN']:
            selected_tactic = 'TACTICAL_TURN'
            selected_maneuver = 'retreat'
        elif allow == ['TACTICAL_EVASION']:
            selected_tactic = 'TACTICAL_EVASION'
            selected_maneuver = 'beam'
        else:
            # 6/7均可：用threat_level阈值区分（可通过环境变量微调）
            try:
                turn_th = float(os.getenv("TACTICAL_DOR_TURN_THRESHOLD", "0.6"))
            except Exception:
                turn_th = 0.6
            if context.threat_level >= turn_th:
                selected_tactic = 'TACTICAL_TURN'
                selected_maneuver = 'retreat'
            else:
                selected_tactic = 'TACTICAL_EVASION'
                selected_maneuver = 'beam'
        
        return {
            'node': 'DOR',
            'tactic': selected_tactic,
            'maneuver': selected_maneuver,
            'defensive': selected_tactic == 'TACTICAL_EVASION'
        }
    
    def decide_at_dr(self, context: Any) -> Dict[str, Any]:
        """
        DR节点决策（65km）
        
        权限：
            - 决定是否重新进攻
            - 10秒时间窗口
        
        Returns:
            决策结果
        """
        agent_id = context.agent_id
        current_time = context.current_time

        # ✅ 文件复盘：DR节点进入（变化触发）
        try:
            from utils.trace_logger import trace_if_changed
            sig = (
                str(getattr(context, "our_intent", None)),
                str(getattr(context, "enemy_intent", None)),
                str(getattr(context, "situation", None)),
                float(getattr(context, "threat_level", 0.0)),
                int(getattr(context, "custom_data", {}).get("missiles_remaining", 0)),
                int(getattr(context, "custom_data", {}).get("enemies_alive", 0)),
            )
            trace_if_changed(
                key=f"node:DR:enter:{agent_id}",
                value=sig,
                标题="DR节点-进入",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(agent_id),
                阶段="DR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="DR：10秒窗口结束后做‘重攻/撤退’二选一，并受团队共识约束",
                数据={
                    "our_intent": str(getattr(context, "our_intent", None)),
                    "enemy_intent": str(getattr(context, "enemy_intent", None)),
                    "situation": str(getattr(context, "situation", None)),
                    "threat_level": float(getattr(context, "threat_level", 0.0)),
                    "enemies_alive": int(getattr(context, "custom_data", {}).get("enemies_alive", 0)),
                    "missiles_remaining": int(getattr(context, "custom_data", {}).get("missiles_remaining", 0)),
                },
            )
        except Exception:
            pass

        # ✅ 防御意图：不进入DR（按用户规则），直接战术回转终止/脱离
        if context.our_intent == 'DEFENSIVE':
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR节点-跳过重攻",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="DENY",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="我方意图=DEFENSIVE：按规则不进入DR重攻，直接TACTICAL_TURN撤离",
                )
            except Exception:
                pass
            return {
                'node': 'DR',
                'tactic': 'TACTICAL_TURN',
                'maneuver': 'retreat',
                'reengage': False,
                'retreat': True
            }

        # ✅ 规则一致性：若已锁定撤退，则不再评估二次进攻
        existing_state = self.agent_dr_decisions.get(agent_id)
        if existing_state and existing_state.get('force_retreat_until', 0) > current_time:
            return {
                'node': 'DR',
                'tactic': 'TACTICAL_TURN',
                'maneuver': 'retreat',
                'reengage': False,
                'retreat': True,
                'retreat_locked': True,
            }
        
        # 🔥 修复1: 为每个飞机单独管理DR窗口状态
        if agent_id not in self.agent_dr_windows:
            self.agent_dr_windows[agent_id] = current_time
            self.agent_dr_decisions[agent_id] = {
                'window_logged': False,
                'window_progress_start_logged': False,
                'window_progress_end_logged': False,
                'force_retreat_until': 0.0,
            }
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR窗口-开始",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="START",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="DR窗口=10秒：窗口内保持防御姿态，窗口结束再做重攻/撤退判断",
                    数据={"window_s": 10.0},
                )
            except Exception:
                pass
        
        # 计算窗口剩余时间
        window_start = self.agent_dr_windows[agent_id]
        window_elapsed = current_time - window_start
        window_remaining = 10.0 - window_elapsed
        
        # 在时间窗口内保持防御姿态
        if window_remaining > 0:
            # 🎯 钳形攻击特殊处理：DR阶段不执行beam机动，而是继续钳形收拢
            current_tactic = context.custom_data.get('current_tactic', None)
            # 🔥 大幅减少窗口日志：每架机最多打印2次（开始一次 + 结束前一次）
            state = self.agent_dr_decisions.get(agent_id, {})
            should_log = False
            if (not state.get('window_progress_start_logged')) and window_remaining > 19.0:
                state['window_progress_start_logged'] = True
                should_log = True
            elif (not state.get('window_progress_end_logged')) and window_remaining < 1.0:
                state['window_progress_end_logged'] = True
                should_log = True
            self.agent_dr_decisions[agent_id] = state

            if current_tactic == 'PINCER_ATTACK':
                if should_log:
                    try:
                        from utils.trace_logger import trace_event
                        trace_event(
                            事件="DR窗口-进行中",
                            env=getattr(context, "env", None),
                            模块="node_decision_methods",
                            类型="DECISION",
                            状态="PERIODIC",
                            我机=str(agent_id),
                            阶段="DR",
                            战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                            说明="窗口内：钳形攻击继续收拢（不做beam以免打断队形）",
                            数据={"window_remaining_s": float(window_remaining)},
                        )
                    except Exception:
                        pass
                return {
                    'node': 'DR',
                    'tactic': None,
                    'maneuver': 'pincer_converge',  # 钳形收拢而非beam
                    'in_window': True,
                    'window_remaining': window_remaining
                }
            else:
                if should_log:
                    try:
                        from utils.trace_logger import trace_event
                        trace_event(
                            事件="DR窗口-进行中",
                            env=getattr(context, "env", None),
                            模块="node_decision_methods",
                            类型="DECISION",
                            状态="PERIODIC",
                            我机=str(agent_id),
                            阶段="DR",
                            战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                            说明="窗口内：保持Beam姿态（等待窗口结束后统一决策）",
                            数据={"window_remaining_s": float(window_remaining)},
                        )
                    except Exception:
                        pass
                return {
                    'node': 'DR',
                    'tactic': None,
                    'maneuver': 'beam',
                    'in_window': True,
                    'window_remaining': window_remaining
                }
        
        # 时间窗口结束，进行决策（避免重复日志）
        agent_decision_state = self.agent_dr_decisions[agent_id]
        if not agent_decision_state.get('window_logged', False):
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR窗口-结束",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="END",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="窗口结束：开始评估重攻/撤退（含规则硬约束+资源/威胁软约束+团队共识）",
                )
            except Exception:
                pass
            agent_decision_state['window_logged'] = True
        
        # 评估条件
        enemies_alive = context.custom_data.get('enemies_alive', 0)
        threat_level = context.threat_level
        fuel_remaining = context.custom_data.get('fuel_remaining', 100)
        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        
        # 决策逻辑：先按“我方意图×态势×敌意图”做硬约束，再叠加威胁/资源软约束
        enemy_type = context.enemy_intent  # ATTACK/NEUTRAL/RETREAT
        enemy_group_phase = str(context.custom_data.get('enemy_group_phase', 'UNKNOWN'))
        enemy_group_pressure_level = int(context.custom_data.get('enemy_group_pressure_level', 0) or 0)
        enemy_target_zone = str(context.custom_data.get('target_zone', 'UNKNOWN')).upper()
        enemy_wave_retreating = bool(context.custom_data.get('enemy_wave_retreating', False))
        enemy_regrouping = enemy_wave_retreating
        if not enemy_regrouping and enemy_group_phase in ('TURN_NORTH', 'REGROUP_NORTH'):
            enemy_regrouping = (
                enemy_target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN')
                or (enemy_target_zone == 'MEDIUM' and enemy_group_pressure_level <= 0)
            )
        enemy_not_pressing = enemy_type in ('RETREAT', 'ESCAPE', 'DEFENSIVE') or enemy_regrouping
        if enemy_type == 'RETREAT':
            enemy_intent_type = 'ESCAPE_TYPE'
        else:
            enemy_intent_type = 'ATTACK_TYPE' if enemy_type == 'ATTACK' else 'NEUTRAL_TYPE'

        # Conservative-clear hard rule: no fresh chase when the opponent is already
        # withdrawing or northbound regrouping.
        rule_allow_reengage = True
        if context.our_intent == 'CONSERVATIVE_CLEAR':
            if context.situation == 'DISADVANTAGE':
                rule_allow_reengage = False
            elif enemy_not_pressing:
                rule_allow_reengage = False
            elif enemies_alive <= 1 and context.situation != 'ADVANTAGE':
                rule_allow_reengage = False

        should_reengage = (
            rule_allow_reengage and
            enemies_alive > 0 and
            threat_level < 0.55 and
            missiles_remaining > 0 and
            context.custom_data.get('team_consensus', True)
        )

        try:
            from utils.trace_logger import trace_if_changed
            sig = (
                bool(rule_allow_reengage),
                int(enemies_alive),
                float(threat_level),
                int(missiles_remaining),
                str(enemy_group_phase),
                bool(enemy_not_pressing),
                bool(context.custom_data.get('team_consensus', True)),
                bool(should_reengage),
            )
            trace_if_changed(
                key=f"dr:eval:{agent_id}",
                value=sig,
                标题="DR决策-条件评估",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(agent_id),
                阶段="DR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="先硬约束(规则)后软约束(威胁/资源/共识)得到是否重攻",
                数据={
                    "rule_allow_reengage": bool(rule_allow_reengage),
                    "enemies_alive": int(enemies_alive),
                    "threat_level": float(threat_level),
                    "missiles_remaining": int(missiles_remaining),
                    "enemy_group_phase": str(enemy_group_phase),
                    "enemy_not_pressing": bool(enemy_not_pressing),
                    "team_consensus_flag": bool(context.custom_data.get('team_consensus', True)),
                    "should_reengage": bool(should_reengage),
                },
            )
        except Exception:
            pass
        
        if should_reengage:
            # 🔥 团队协调机制：确保所有己方飞机统一决策
            team_prefix = agent_id[0]  # A或B
            consensus_key = f"{team_prefix}_DR_consensus"
            
            # 检查团队共识
            if not self._check_team_consensus(team_prefix, 'reengage', context):
                try:
                    from utils.trace_logger import trace_throttle
                    trace_throttle(
                        key=f"dr:wait_consensus:reengage:{agent_id}",
                        min_steps=50,
                        标题="DR决策-等待共识",
                        env=getattr(context, "env", None),
                        模块="node_decision_methods",
                        类型="DECISION",
                        状态="DENY",
                        我机=str(agent_id),
                        阶段="DR",
                        战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                        说明="个人倾向重攻，但团队共识尚未达成 → 继续beam等待",
                    )
                except Exception:
                    pass
                # 暂时保持当前状态，等待团队决策统一
                return {
                    'node': 'DR',
                    'tactic': 'BEAM',  # 暂时保持Beam机动
                    'maneuver': 'continue_beam',
                    'reengage': False,
                    'waiting_consensus': True
                }
            
            # 重新进攻 - 修改为正确触发战术回转
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR决策-重攻",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="OK",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="达成重攻：将触发TACTICAL_TURN(reengage_turn)并进入二次进攻",
                    数据={"enemies_alive": int(enemies_alive), "threat": float(threat_level)},
                )
            except Exception:
                pass
            
            # 选择第二轮战术（严格走DR决策表，使“我方意图差异”真实生效）
            if self.decision_table:
                candidates = self.decision_table.query_candidates('DR', context.our_intent, context.enemy_intent, context.situation)
            else:
                candidates = ['DRAG_SHOOT']
            selected_tactic = candidates[0] if candidates else 'DRAG_SHOOT'

            # ✅ 文件复盘：DR决策表候选→二次进攻模板选择
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR决策-二次进攻模板",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="TACTIC",
                    状态="RETURN",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="DR决策表查询候选模板并选择默认最优（用于复盘二次进攻策略来源）",
                    数据={"candidates": list(candidates) if isinstance(candidates, (list, tuple)) else None, "selected": str(selected_tactic)},
                )
            except Exception:
                pass
            
            # 正确触发战术回转
            decision_result = {
                'node': 'DR',
                'tactic': 'TACTICAL_TURN',  # 明确指定战术回转
                'maneuver': 'TACTICAL_TURN',  # 设置战术回转机动
                'turn_type': 'reengage_turn',  # 重新交战回转
                'reengage': True,
                'is_second_attack': True,
                'target_heading': 'toward_enemy',  # 明确指定朝向敌机
                'second_attack_tactic': selected_tactic  # 保留选中的战术
            }
            
            # 保存决策结果供战术执行器使用
            self.last_decision = decision_result
            
            return decision_result
        else:
            # ✅ 修复：保守+敌方占优按规则应直接终止/撤退，不应“等待共识”卡住
            if not rule_allow_reengage:
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="DR决策-撤退(规则)",
                        env=getattr(context, "env", None),
                        模块="node_decision_methods",
                        类型="DECISION",
                        状态="DENY",
                        我机=str(agent_id),
                        阶段="DR",
                        战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                        说明="保守肃清+敌方占优：按规则硬约束直接撤退(TACTICAL_TURN)",
                    )
                except Exception:
                    pass
                state = self.agent_dr_decisions.get(agent_id, {})
                state['force_retreat_until'] = current_time + 30.0
                self.agent_dr_decisions[agent_id] = state
                return {
                    'node': 'DR',
                    'tactic': 'TACTICAL_TURN',
                    'maneuver': 'retreat',
                    'reengage': False,
                    'retreat': True
                }

            # 🔥 团队协调机制：确保所有己方飞机统一撤退决策
            team_prefix = agent_id[0]  # A或B
            
            # 检查团队撤退共识
            if not self._check_team_consensus(team_prefix, 'retreat', context):
                # ✅ 修复：如果无法达成撤退共识，也不能卡死在“等待”里（会导致窗口结束后一直beam）
                # 兜底：激进更偏重攻，保守/防御更偏撤退
                if context.our_intent == 'AGGRESSIVE_CLEAR':
                    try:
                        from utils.trace_logger import trace_throttle
                        trace_throttle(
                            key=f"dr:fallback:aggressive:{agent_id}",
                            min_steps=80,
                            标题="DR决策-兜底",
                            env=getattr(context, "env", None),
                            模块="node_decision_methods",
                            类型="DECISION",
                            状态="FAILSAFE",
                            我机=str(agent_id),
                            阶段="DR",
                            战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                            说明="撤退共识未达成：激进意图兜底保持beam姿态",
                        )
                    except Exception:
                        pass
                    return {'node': 'DR', 'tactic': None, 'maneuver': 'beam', 'reengage': False, 'waiting_consensus': True}
                try:
                    from utils.trace_logger import trace_throttle
                    trace_throttle(
                        key=f"dr:fallback:retreat:{agent_id}",
                        min_steps=80,
                        标题="DR决策-兜底",
                        env=getattr(context, "env", None),
                        模块="node_decision_methods",
                        类型="DECISION",
                        状态="FAILSAFE",
                        我机=str(agent_id),
                        阶段="DR",
                        战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                        说明="撤退共识未达成：非激进意图兜底执行撤退(TACTICAL_TURN)",
                    )
                except Exception:
                    pass
                state = self.agent_dr_decisions.get(agent_id, {})
                state['force_retreat_until'] = current_time + 30.0
                self.agent_dr_decisions[agent_id] = state
                return {'node': 'DR', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'reengage': False, 'retreat': True}
            
            # 撤退
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="DR决策-撤退",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="OK",
                    我机=str(agent_id),
                    阶段="DR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="DR评估后选择撤退：TACTICAL_TURN（RTB语义锁存）",
                    数据={"enemies_alive": int(enemies_alive), "threat": float(threat_level)},
                )
            except Exception:
                pass
            state = self.agent_dr_decisions.get(agent_id, {})
            state['force_retreat_until'] = current_time + 30.0
            self.agent_dr_decisions[agent_id] = state
            return {
                'node': 'DR',
                'tactic': 'TACTICAL_TURN',
                'maneuver': 'retreat',
                'reengage': False,
                'retreat': True
            }
    
    def _check_team_consensus(self, team_prefix: str, decision_type: str, context: Any) -> bool:
        """
        检查团队决策共识
        
        Args:
            team_prefix: 团队前缀 ('A' 或 'B')
            decision_type: 决策类型 ('reengage' 或 'retreat')
            context: 决策上下文
            
        Returns:
            bool: 是否达成共识
        """
        current_time = getattr(context, 'current_time', 0)
        consensus_key = f"{team_prefix}_{decision_type}_{current_time//10}"  # 每10秒检查一次共识
        
        # 如果已经有共识缓存，直接返回
        if consensus_key in self.decision_consensus_cache:
            return self.decision_consensus_cache[consensus_key]
        
        # 模拟团队决策评估（在实际实现中，这里应该检查所有团队成员的意向）
        team_agents = [f"{team_prefix}0100", f"{team_prefix}0200"]  # 长机和僚机
        
        # 基于威胁水平和资源状况进行团队决策
        threat_level = context.threat_level
        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        fuel_remaining = context.custom_data.get('fuel_remaining', 100)
        enemies_alive = context.custom_data.get('enemies_alive', 0)
        
        # 团队统一决策逻辑
        if decision_type == 'reengage':
            # 重新进攻的共识条件（放宽威胁阈值，使二次进攻更容易触发）
            cond_enemies = enemies_alive > 0
            cond_threat = threat_level < 0.50  # 🔧 放宽威胁阈值（从0.35提高到0.50）
            cond_missiles = missiles_remaining > 0
            consensus = cond_enemies and cond_threat and cond_missiles
            # ✅ 文件复盘：团队共识评估（节流）
            try:
                from utils.trace_logger import trace_throttle

                trace_throttle(
                    key=f"dr:consensus:{team_prefix}:reengage:{int(current_time//10)}",
                    min_steps=50,
                    标题="DR共识-重新进攻",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="ALLOW" if consensus else "DENY",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段="DR",
                    说明="团队共识：敌机存活+威胁阈值+剩余导弹 → 是否同意重攻",
                    数据={
                        "team": str(team_prefix),
                        "enemies_alive": int(enemies_alive),
                        "cond_enemies": bool(cond_enemies),
                        "threat": float(threat_level),
                        "th_th": 0.50,
                        "cond_threat": bool(cond_threat),
                        "missiles": int(missiles_remaining),
                        "cond_missiles": bool(cond_missiles),
                        "consensus": bool(consensus),
                    },
                )
            except Exception:
                pass
        else:  # retreat
            # 撤退的共识条件（删除燃料判断）
            cond_no_enemies = enemies_alive == 0
            cond_high_threat = threat_level > 0.6
            cond_no_missiles = missiles_remaining == 0
            consensus = cond_no_enemies or cond_high_threat or cond_no_missiles
            try:
                from utils.trace_logger import trace_throttle

                trace_throttle(
                    key=f"dr:consensus:{team_prefix}:retreat:{int(current_time//10)}",
                    min_steps=50,
                    标题="DR共识-撤退",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="ALLOW" if consensus else "DENY",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段="DR",
                    说明="团队共识：无敌机/高威胁/无导弹 → 是否同意撤退",
                    数据={
                        "team": str(team_prefix),
                        "enemies_alive": int(enemies_alive),
                        "cond_no_enemies": bool(cond_no_enemies),
                        "threat": float(threat_level),
                        "cond_high_threat": bool(cond_high_threat),
                        "missiles": int(missiles_remaining),
                        "cond_no_missiles": bool(cond_no_missiles),
                        "consensus": bool(consensus),
                    },
                )
            except Exception:
                pass
        
        # 缓存决策结果（避免频繁计算）
        self.decision_consensus_cache[consensus_key] = consensus
        
        return consensus
    
    # ==================== 三级节点决策（机动级） ====================
    
    def decide_at_mtr(self, context: Any) -> Dict[str, Any]:
        """
        MTR节点决策（80km）
        
        权限：
            - 只能选择机动
            - 不能切换战术
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:MTR:{context.agent_id}",
                min_steps=120,
                标题="MTR节点-机动决策",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="MTR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="三级节点：不切换模板，仅做温和占位/航迹调整",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass
        
        # 计算温和的攻击占位点（小幅度调整）
        try:
            my_ac = context.env.agents.get(context.agent_id)
            tgt_id = context.env.target_map.get(context.agent_id) if hasattr(context.env, 'target_map') else None
            if not tgt_id:
                from tactical_types import get_target_with_fallback
                tgt_id = get_target_with_fallback(context.agent_id, context.env)
            tgt_ac = context.env._jsbsims.get(tgt_id) if tgt_id else None
            if my_ac and tgt_ac and tgt_ac.is_alive:
                my_pos = my_ac.get_position()
                tgt_pos = tgt_ac.get_position()
                los = tgt_pos - my_pos
                los[:2] = los[:2] / (np.linalg.norm(los[:2]) + 1e-6)
                # 前向8km占位，侧向微偏移2km（根据机号区分左右）
                forward = 8000.0
                lateral = 2000.0 if context.agent_id.endswith('100') else -2000.0
                perp = np.array([ -los[1], los[0], 0.0 ])
                wp = my_pos + los * forward + perp * (lateral / (np.linalg.norm(perp[:2]) + 1e-6))
                wp[2] = my_pos[2]
            else:
                wp = None
        except Exception:
            wp = None
        
        return {
            'node': 'MTR',
            'tactic': None,
            'maneuver': 'straight',
            'attack_waypoint': wp.tolist() if wp is not None else None
        }
    
    def decide_at_tr(self, context: Any) -> Dict[str, Any]:
        """
        TR节点决策（75km）
        
        权限：
            - 准备规避机动
            - 中制导结束
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:TR:{context.agent_id}",
                min_steps=120,
                标题="TR节点-机动决策",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="TR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="TR：默认准备规避；防御意图可触发直接脱离",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass
        
        # ✅ 防御意图在TR节点的“终止/脱离”规则（按用户表述）
        if context.our_intent == 'DEFENSIVE':
            if (context.situation != 'ADVANTAGE') or (context.enemy_intent in ('RETREAT', 'ESCAPE')):
                try:
                    from utils.trace_logger import trace_event
                    trace_event(
                        事件="TR节点-脱离",
                        env=getattr(context, "env", None),
                        模块="node_decision_methods",
                        类型="DECISION",
                        状态="OK",
                        我机=str(getattr(context, "agent_id", None)),
                        阶段="TR",
                        战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                        说明="防御意图 + 非优势/敌撤退：触发TACTICAL_TURN脱离",
                    )
                except Exception:
                    pass
                return {'node': 'TR', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat'}

        # 默认：TR阶段准备规避
        return {'node': 'TR', 'tactic': None, 'maneuver': 'prepare_evasion'}
    
    def decide_at_mar(self, context: Any) -> Dict[str, Any]:
        """
        MAR节点决策（40km）
        
        权限：
            - 强制防御机动
            - 准备近距格斗或脱离
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:MAR:{context.agent_id}",
                min_steps=120,
                标题="MAR节点-近距决策",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="MAR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="MAR：通常强制防御；若正在编队重整/二次进攻则不强制切换",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass
        
        # 🔥 关键修复：检查是否正在执行二次进攻/队形重置
        # 如果正在进行二次进攻，不强制切换到防御战术
        current_tactic = context.custom_data.get('current_tactic', None)
        is_second_attack = context.is_second_attack
        
        if current_tactic == 'TACTICAL_TURN':
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="MAR节点-跳过强制防御",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="DENY",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段="MAR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="编队重整/撤退进行中：不强制切换防御",
                )
            except Exception:
                pass
            return {
                'node': 'MAR',
                'tactic': None,  # 不切换战术
                'maneuver': 'continue_formation_reset',
                'force_defensive': False
            }
        
        if is_second_attack:
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="MAR节点-跳过强制防御",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="DENY",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段="MAR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="二次进攻进行中：不强制切换防御",
                )
            except Exception:
                pass
            return {
                'node': 'MAR',
                'tactic': None,  # 不切换战术
                'maneuver': 'continue_second_attack',
                'force_defensive': False
            }

        # ✅ 防御意图：在MAR节点不进入Beam战术规避，直接脱离/回转
        # 真实来袭导弹规避由更高优先级的导弹规避逻辑接管（executor._check_and_evade_missile）
        if context.our_intent == 'DEFENSIVE':
            try:
                from utils.trace_logger import trace_event
                trace_event(
                    事件="MAR节点-脱离",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="DECISION",
                    状态="OK",
                    我机=str(getattr(context, "agent_id", None)),
                    阶段="MAR",
                    战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                    说明="防御意图：MAR直接TACTICAL_TURN脱离（导弹来袭规避由更高优先级接管）",
                )
            except Exception:
                pass
            return {
                'node': 'MAR',
                'tactic': 'TACTICAL_TURN',
                'maneuver': 'retreat',
                'force_defensive': True
            }
        
        # MAR强制防御（常规情况）
        return {
            'node': 'MAR',
            'tactic': 'TACTICAL_EVASION',
            'maneuver': 'defensive',
            'force_defensive': True
        }
    
    # ==================== 四级节点决策（参数级） ====================
    
    def decide_at_lr(self, context: Any) -> Dict[str, Any]:
        """
        LR节点决策（78km）
        
        权限：
            - 只能调整参数
            - 发射导弹决策
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:LR:{context.agent_id}",
                min_steps=120,
                标题="LR节点-参数/发射决策",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="LR",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="四级节点：参数调整 + 是否请求发射（不切换模板）",
                数据={
                    "distance_km": float(getattr(context, "distance", 0.0)) / 1000.0,
                    "threat": float(getattr(context, "threat_level", 0.0)),
                },
            )
        except Exception:
            pass
        
        # 决定是否发射导弹
        should_launch = self._evaluate_launch_condition(context)
        
        # 决定是否执行Crank机动
        should_crank = context.threat_level > 0.3
        
        return {
            'node': 'LR',
            'tactic': None,
            'maneuver': 'crank' if should_crank else 'straight',
            'launch_missile': should_launch,
            'lr_maneuver': 'crank' if should_crank else 'straight'
        }
    
    # ==================== 第二轮攻击节点决策 ====================
    
    def decide_at_mtr2(self, context: Any) -> Dict[str, Any]:
        """
        MTR2节点决策（第二轮80km）
        
        特点：
            - 更谨慎的决策
            - 考虑剩余资源
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:MTR2:{context.agent_id}",
                min_steps=120,
                标题="MTR2节点-第二轮",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="MTR2",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="第二轮：更谨慎；防御不进入，保守非优势终止",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass
        
        # ✅ 按用户规则：防御意图不进入第二轮；保守在敌方占优时终止
        if context.our_intent == 'DEFENSIVE':
            return {'node': 'MTR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': False}
        if context.our_intent == 'CONSERVATIVE_CLEAR' and context.situation != 'ADVANTAGE':
            return {'node': 'MTR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': True}

        # 默认：维持二次进攻模板执行（具体动作仍由各战术模板序列控制）
        return {'node': 'MTR2', 'tactic': None, 'maneuver': 'continue_second_attack', 'is_second_attack': True}
    
    def decide_at_lr2(self, context: Any) -> Dict[str, Any]:
        """
        LR2节点决策（第二轮78km）
        
        特点：
            - 谨慎发射
            - 保留最后导弹
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:LR2:{context.agent_id}",
                min_steps=120,
                标题="LR2节点-第二轮发射",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="LR2",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="第二轮：是否继续请求发射（最终火控仍会再约束）",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass
        
        if context.our_intent == 'DEFENSIVE':
            return {'node': 'LR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': False}
        if context.our_intent == 'CONSERVATIVE_CLEAR' and context.situation != 'ADVANTAGE':
            return {'node': 'LR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': True}

        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        should_launch = missiles_remaining > 0  # 发射策略由模板内部/火控再约束
        return {'node': 'LR2', 'tactic': None, 'maneuver': 'continue_second_attack', 'launch_missile': should_launch, 'is_second_attack': True}
    
    def decide_at_tr2(self, context: Any) -> Dict[str, Any]:
        """
        TR2节点决策（第二轮75km）
        
        特点：
            - 准备脱离
            - 最大规避
        
        Returns:
            决策结果
        """
        try:
            from utils.trace_logger import trace_throttle
            trace_throttle(
                key=f"node:TR2:{context.agent_id}",
                min_steps=120,
                标题="TR2节点-第二轮",
                env=getattr(context, "env", None),
                模块="node_decision_methods",
                类型="DECISION",
                状态="INPUT",
                我机=str(getattr(context, "agent_id", None)),
                阶段="TR2",
                战术=str(getattr(context, "custom_data", {}).get("current_tactic", None)),
                说明="第二轮：准备脱离/规避；防御不进入，保守非优势终止",
                数据={"distance_km": float(getattr(context, "distance", 0.0)) / 1000.0},
            )
        except Exception:
            pass

        # ✅ 按用户规则：防御意图不进入第二轮；保守在敌方占优时终止
        if context.our_intent == 'DEFENSIVE':
            return {'node': 'TR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': False}
        if context.our_intent == 'CONSERVATIVE_CLEAR' and context.situation != 'ADVANTAGE':
            return {'node': 'TR2', 'tactic': 'TACTICAL_TURN', 'maneuver': 'retreat', 'retreat': True, 'is_second_attack': True}

        # 默认：二次进攻继续（具体动作仍由模板内部序列控制；是否返航由二次进攻执行逻辑控制）
        return {'node': 'TR2', 'tactic': None, 'maneuver': 'continue_second_attack', 'is_second_attack': True}
    
    # ==================== 辅助方法 ====================
    
    def _determine_formation_adjustment(self, tactic: str) -> Dict[str, Any]:
        """
        根据战术确定编队调整参数
        
        Args:
            tactic: 战术名称
        
        Returns:
            编队调整参数
        """
        formation_map = {
            'DRAG_SHOOT': {'type': 'trail', 'distance': 8000},
            'PINCER_ATTACK': {'type': 'spread', 'distance': 10000},
            'HIGH_LOW_ATTACK': {'type': 'vertical', 'altitude_diff': 3000},
            'FRONT_BACK': {'type': 'trail', 'distance': 5500},
            'SIDE_BY_SIDE': {'type': 'line', 'distance': 6000},
        }
        return formation_map.get(tactic, {'type': 'default'})
    
    def _get_meld_formation_adjustment(self, tactic: str) -> Dict[str, Any]:
        """
        获取MELD阶段的编队调整参数
        
        Args:
            tactic: 战术名称
        
        Returns:
            编队调整参数
        """
        meld_formation_map = {
            'DRAG_SHOOT': {'type': 'converge', 'angle': 0},
            'PINCER_ATTACK': {'type': 'maintain_spread', 'angle': 15},
            'HIGH_LOW_ATTACK': {'type': 'altitude_adjust', 'target_alt_diff': 3000},
            'FRONT_BACK': {'type': 'establish_trail', 'target_distance': 5500},
            'SIDE_BY_SIDE': {'type': 'parallel', 'lateral_distance': 6000},
        }
        return meld_formation_map.get(tactic, {'type': 'maintain'})
    
    def _check_tactic_switch_condition(self, context: Any) -> bool:
        """
        检查是否满足战术切换条件
        
        Args:
            context: 决策上下文
        
        Returns:
            是否需要切换战术
        """
        # 检查各种切换条件
        threat_changed = abs(context.threat_level - 
                           context.custom_data.get('last_threat', 0)) > 0.3
        enemies_changed = (context.custom_data.get('enemies_alive', 2) != 
                         context.custom_data.get('last_enemies', 2))
        situation_changed = (context.situation != 
                           context.custom_data.get('last_situation', ''))
        
        return threat_changed or enemies_changed or situation_changed
    
    def _evaluate_launch_condition(self, context: Any) -> bool:
        """
        评估导弹发射条件 - 🔥 修复：确保长机和僚机都能发射
        
        Args:
            context: 决策上下文
        
        Returns:
            是否应该发射导弹
        """
        # ✅ 修复：原逻辑依赖 custom_data.has_radar_lock / aspect_angle，但实际未稳定提供 → 导致LR几乎从不"请求发射"
        # 策略：节点层只做"强烈建议发射"的请求（LR/LR'高概率），最终是否能发射仍由
        # tactical_task._handle_missile_launches + missile_manager.should_launch_missile + 雷达系统严格把关。
        
        # 🔥 修复僚机发射问题：检查导弹数量（对长机和僚机都适用）
        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        if missiles_remaining <= 0:
            return False

        d = float(context.distance)
        agent_id = context.agent_id
        
        # 🔥 修复：确保长机和僚机都能在LR节点发射
        # 首轮LR：60~85km为主要窗口（对长机和僚机都适用）
        if context.is_second_attack:
            # LR'≈53km（用户要求），给一点余量
            launch_window = 52000 <= d <= 90000
        else:
            # 首轮LR：60~85km为主要窗口
            launch_window = 60000 <= d <= 85000
        
        # 🔥 关键修复：对僚机也允许发射（之前可能被误判为不允许）
        if launch_window:
            try:
                from utils.trace_logger import trace_throttle
                trace_throttle(
                    key=f"lr_launch_suggest:{agent_id}",
                    min_steps=120,
                    标题="LR发射评估-建议发射",
                    env=getattr(context, "env", None),
                    模块="node_decision_methods",
                    类型="GATE",
                    状态="ALLOW",
                    我机=str(agent_id),
                    阶段="LR2" if bool(getattr(context, "is_second_attack", False)) else "LR",
                    说明="节点层只发出‘建议发射’请求；最终是否发射由火控/雷达/导弹管理器把关",
                    数据={"distance_km": float(d) / 1000.0, "missiles_remaining": int(missiles_remaining), "is_second_attack": bool(getattr(context, "is_second_attack", False))},
                )
            except Exception:
                pass
            return True
        
        return False
