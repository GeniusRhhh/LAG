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


class NodeDecisionMethods:
    """
    节点决策方法集合
    包含各控制距离节点的具体决策实现
    """
    
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
            logging.warning(f"未知节点: {node}")
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
        logging.info(f"⚡ [NLT决策] 距离{context.distance/1000:.0f}km")
        
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
        
        logging.info(f"   → NLT决策: 战术={selected_tactic}, 编队调整={formation_adjustment}")
        
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
        logging.info(f"⚡ [MELD决策] 距离{context.distance/1000:.0f}km")
        
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
        
        logging.info(f"   → MELD决策: 战术切换={need_switch}, 编队={formation_adjustment}")
        
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
        logging.info(f"⚡ [DOR决策] 距离{context.distance/1000:.0f}km")
        
        # 1. 威胁评估
        high_threat = context.threat_level > 0.7
        
        # 2. 决定是否切换到防御
        if high_threat:
            selected_tactic = 'TACTICAL_EVASION'
            selected_maneuver = 'beam'
            logging.warning(f"   ⚠️ 高威胁({context.threat_level:.2f})，切换到战术规避")
        else:
            selected_tactic = None
            selected_maneuver = 'straight'
        
        return {
            'node': 'DOR',
            'tactic': selected_tactic,
            'maneuver': selected_maneuver,
            'defensive': high_threat
        }
    
    def decide_at_dr(self, context: Any) -> Dict[str, Any]:
        """
        DR节点决策（65km）
        
        权限：
            - 决定是否重新进攻
            - 20秒时间窗口
        
        Returns:
            决策结果
        """
        current_time = context.current_time
        
        # 初始化DR窗口
        if self.dr_window_start is None:
            self.dr_window_start = current_time
            logging.info(f"⚡ [DR决策] 开始20秒决策窗口")
        
        # 计算窗口剩余时间
        window_elapsed = current_time - self.dr_window_start
        window_remaining = 20.0 - window_elapsed
        
        # 在时间窗口内保持防御姿态
        if window_remaining > 0:
            logging.info(f"   → DR窗口: 剩余{window_remaining:.1f}秒，保持Beam姿态")
            return {
                'node': 'DR',
                'tactic': None,
                'maneuver': 'beam',
                'in_window': True,
                'window_remaining': window_remaining
            }
        
        # 时间窗口结束，进行决策
        logging.info(f"⚡ [DR决策] 时间窗口结束，评估是否重新进攻")
        
        # 评估条件
        enemies_alive = context.custom_data.get('enemies_alive', 0)
        threat_level = context.threat_level
        fuel_remaining = context.custom_data.get('fuel_remaining', 100)
        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        
        # 决策逻辑
        should_reengage = (
            enemies_alive > 0 and
            threat_level < 0.6 and
            fuel_remaining > 30 and
            missiles_remaining > 0
        )
        
        if should_reengage:
            # 重新进攻
            logging.info(f"   → 决定重新进攻！敌机{enemies_alive}架，威胁{threat_level:.2f}")
            
            # 选择第二轮战术
            if self.decision_table:
                candidates = self.decision_table.query_candidates(
                    'DR',
                    context.our_intent,
                    context.enemy_intent,
                    context.situation
                )
            else:
                candidates = ['PINCER_ATTACK', 'SIDE_BY_SIDE']
            
            selected_tactic = candidates[0] if candidates else 'PINCER_ATTACK'
            
            return {
                'node': 'DR',
                'tactic': selected_tactic,
                'maneuver': 'turn',
                'reengage': True,
                'is_second_attack': True
            }
        else:
            # 撤退
            logging.info(f"   → 决定撤退。敌机{enemies_alive}架，威胁{threat_level:.2f}")
            return {
                'node': 'DR',
                'tactic': 'TACTICAL_TURN',
                'maneuver': 'retreat',
                'reengage': False,
                'retreat': True
            }
    
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
        logging.info(f"⚡ [MTR决策] 距离{context.distance/1000:.0f}km")
        
        # 根据当前战术选择合适的机动
        current_tactic = context.custom_data.get('current_tactic', 'PINCER_ATTACK')
        
        # MTR阶段机动选择（简化处理）
        selected_maneuver = 'straight'  # 默认保持直线飞行
        
        return {
            'node': 'MTR',
            'tactic': None,  # 不改变战术
            'maneuver': selected_maneuver
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
        logging.info(f"⚡ [TR决策] 距离{context.distance/1000:.0f}km")
        
        # TR阶段通常需要准备规避
        return {
            'node': 'TR',
            'tactic': None,
            'maneuver': 'prepare_evasion'
        }
    
    def decide_at_mar(self, context: Any) -> Dict[str, Any]:
        """
        MAR节点决策（40km）
        
        权限：
            - 强制防御机动
            - 准备近距格斗或脱离
        
        Returns:
            决策结果
        """
        logging.info(f"⚡ [MAR决策] 距离{context.distance/1000:.0f}km，进入近距")
        
        # MAR强制防御
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
        logging.info(f"⚡ [LR决策] 距离{context.distance/1000:.0f}km")
        
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
        logging.info(f"⚡ [MTR2决策] 第二轮攻击，距离{context.distance/1000:.0f}km")
        
        # 第二轮更谨慎，倾向防御性机动
        return {
            'node': 'MTR2',
            'tactic': None,
            'maneuver': 'defensive_approach',
            'is_second_attack': True
        }
    
    def decide_at_lr2(self, context: Any) -> Dict[str, Any]:
        """
        LR2节点决策（第二轮78km）
        
        特点：
            - 谨慎发射
            - 保留最后导弹
        
        Returns:
            决策结果
        """
        logging.info(f"⚡ [LR2决策] 第二轮发射，距离{context.distance/1000:.0f}km")
        
        missiles_remaining = context.custom_data.get('missiles_remaining', 0)
        
        # 只有在条件很好时才发射最后的导弹
        should_launch = missiles_remaining > 0 and context.threat_level < 0.4
        
        return {
            'node': 'LR2',
            'tactic': None,
            'maneuver': 'defensive_crank',
            'launch_missile': should_launch,
            'is_second_attack': True
        }
    
    def decide_at_tr2(self, context: Any) -> Dict[str, Any]:
        """
        TR2节点决策（第二轮75km）
        
        特点：
            - 准备脱离
            - 最大规避
        
        Returns:
            决策结果
        """
        logging.info(f"⚡ [TR2决策] 第二轮规避，距离{context.distance/1000:.0f}km")
        
        # 第二轮TR后准备脱离
        return {
            'node': 'TR2',
            'tactic': 'TACTICAL_TURN',
            'maneuver': 'max_evasion',
            'prepare_disengage': True,
            'is_second_attack': True
        }
    
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
        评估导弹发射条件
        
        Args:
            context: 决策上下文
        
        Returns:
            是否应该发射导弹
        """
        # 简化的发射条件评估
        has_lock = context.custom_data.get('has_radar_lock', False)
        in_range = context.distance < 78000  # LR距离
        good_angle = context.custom_data.get('aspect_angle', 0) < 60
        
        return has_lock and in_range and good_angle
