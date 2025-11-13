"""
完整智能战术选择系统 - 实现项目说明.md的完整算法

整合：
- 态势评估（6.2节）
- 威胁评估（6.3节）
- 意图识别（6.4节）
- 决策表查询（6.5.6节）
- 战术选择算法（Algorithm 6-1）
- 战术适应性评估（Algorithm 6-1.1）
"""
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional

# 导入新实现的模块
from .threat_evaluator_complete import CompleteThreatEvaluator
from .intent_recognizer import IntentRecognizer, EnemyIntent, FriendlyIntent
from .decision_table import DecisionTable
from .tactical_selector_algorithm import TacticalSelectorAlgorithm


class CompleteTacticalSystem:
    """完整智能战术选择系统"""
    
    def __init__(self, my_intent: str = 'CONSERVATIVE_CLEAR'):
        """
        初始化完整战术系统
        
        Args:
            my_intent: 我方意图 ('AGGRESSIVE_CLEAR', 'CONSERVATIVE_CLEAR', 'DEFENSIVE')
        """
        # 我方意图
        self.my_intent = my_intent
        
        # 初始化子系统
        self.threat_evaluator = CompleteThreatEvaluator()
        self.intent_recognizer = IntentRecognizer()
        self.decision_table = DecisionTable()
        self.tactical_selector = TacticalSelectorAlgorithm(
            self.threat_evaluator,
            self.decision_table
        )
        
        # 当前战术状态
        self.current_tactic = None
        self.current_roles = {}
        
        logging.info(f"✅ 完整智能战术选择系统初始化完成 (我方意图: {my_intent})")
    
    def select_tactic(self, control_distance, my_aircraft_list, enemy_aircraft_list, env):
        """
        选择战术模板
        
        Args:
            control_distance: 控制距离 ('NLT', 'MELD', 'DOR', 'DR', etc.)
            my_aircraft_list: 我方飞机列表 [长机, 僚机]
            enemy_aircraft_list: 敌方飞机列表
            env: 环境对象
            
        Returns:
            tuple: (tactic_name, roles)
        """
        try:
            # 1. 态势评估
            situation = self._evaluate_situation(my_aircraft_list, enemy_aircraft_list)
            
            # 2. 意图识别
            enemy_intent = self._recognize_enemy_intent(my_aircraft_list, enemy_aircraft_list)
            
            # 3. 战术选择（Algorithm 6-1）
            selected_tactic = self.tactical_selector.select_tactic(
                control_distance=control_distance,
                my_intent=self.my_intent,
                enemy_intent=enemy_intent,
                situation=situation,
                my_aircraft=my_aircraft_list,
                enemy_aircraft=enemy_aircraft_list,
                env=env,
                current_tactic=self.current_tactic
            )
            
            # 4. 角色分配
            roles = self._assign_roles(selected_tactic, my_aircraft_list, enemy_aircraft_list, env)
            
            # 更新当前状态
            self.current_tactic = selected_tactic
            self.current_roles = roles
            
            logging.info(f"🎯 战术选择完成: {selected_tactic} | 角色: {roles}")
            
            return selected_tactic, roles
            
        except Exception as e:
            logging.error(f"战术选择错误: {e}")
            return 'SIDE_BY_SIDE', {'lead': 'left', 'wingman': 'right'}
    
    def _evaluate_situation(self, my_aircraft_list, enemy_aircraft_list):
        """
        评估战场态势
        
        Returns:
            str: 'ADVANTAGE', 'NEUTRAL', 'DISADVANTAGE'
        """
        try:
            # 简化版本：基于数量和存活状态
            my_alive = sum(1 for ac in my_aircraft_list if ac and ac.is_alive)
            enemy_alive = sum(1 for ac in enemy_aircraft_list if ac and ac.is_alive)
            
            if my_alive > enemy_alive:
                return 'ADVANTAGE'
            elif my_alive < enemy_alive:
                return 'DISADVANTAGE'
            else:
                # 数量相等，基于高度和速度
                if my_aircraft_list and enemy_aircraft_list:
                    my_lead = my_aircraft_list[0] if my_aircraft_list[0].is_alive else my_aircraft_list[1]
                    enemy_lead = enemy_aircraft_list[0] if enemy_aircraft_list[0].is_alive else enemy_aircraft_list[1]
                    
                    my_alt = my_lead.get_position()[2]
                    enemy_alt = enemy_lead.get_position()[2]
                    
                    if my_alt - enemy_alt > 1000:
                        return 'ADVANTAGE'
                    elif enemy_alt - my_alt > 1000:
                        return 'DISADVANTAGE'
                
                return 'NEUTRAL'
                
        except Exception as e:
            logging.error(f"态势评估错误: {e}")
            return 'NEUTRAL'
    
    def _recognize_enemy_intent(self, my_aircraft_list, enemy_aircraft_list):
        """
        识别敌方意图
        
        Returns:
            str: 'ATTACK', 'NEUTRAL', 'RETREAT'
        """
        try:
            if not my_aircraft_list or not enemy_aircraft_list:
                return 'NEUTRAL'
            
            # 使用意图识别器
            my_lead = my_aircraft_list[0] if my_aircraft_list[0].is_alive else my_aircraft_list[1]
            enemy_lead = enemy_aircraft_list[0] if enemy_aircraft_list[0].is_alive else enemy_aircraft_list[1]
            
            intent = self.intent_recognizer.recognize_enemy_intent(my_lead, enemy_lead)
            
            return intent
            
        except Exception as e:
            logging.error(f"意图识别错误: {e}")
            return 'NEUTRAL'

    def _assign_roles(self, tactic, my_aircraft_list, enemy_aircraft_list, env):
        """
        分配战术角色（简化版本 - 固定1v1对应关系）

        固定对应关系：
        - A0100 (我方长机) → B0100 (敌方长机)
        - A0200 (我方僚机) → B0200 (敌方僚机)

        不再根据战术类型动态调整角色，所有战术都使用固定的1v1对应关系。

        Args:
            tactic: 战术名称
            my_aircraft_list: 我方飞机列表
            enemy_aircraft_list: 敌方飞机列表
            env: 环境对象

        Returns:
            dict: 固定角色分配
        """
        # 固定角色分配：不再根据战术动态调整
        # A0100始终是长机，A0200始终是僚机
        # A0100对抗B0100，A0200对抗B0200
        return {
            'lead': 'A0100',           # 长机ID
            'wingman': 'A0200',        # 僚机ID
            'lead_target': 'B0100',    # 长机目标（固定）
            'wingman_target': 'B0200', # 僚机目标（固定）
            'tactic': tactic           # 当前战术
        }

    def get_threat_level(self, my_aircraft, enemy_aircraft_list, env):
        """
        获取威胁等级

        Args:
            my_aircraft: 我方飞机
            enemy_aircraft_list: 敌方飞机列表
            env: 环境对象

        Returns:
            float: 威胁等级 [0, 1]
        """
        try:
            if not enemy_aircraft_list:
                return 0.0

            # 计算对所有敌机的威胁
            max_threat = 0.0
            for enemy_ac in enemy_aircraft_list:
                if enemy_ac and enemy_ac.is_alive:
                    threat = self.threat_evaluator.calculate_total_threat(my_aircraft, enemy_ac, env)
                    max_threat = max(max_threat, threat)

            return max_threat

        except Exception as e:
            logging.error(f"威胁等级计算错误: {e}")
            return 0.5

    def check_emergency_interrupt(self, my_aircraft, enemy_aircraft_list, env):
        """
        检查紧急中断条件

        Returns:
            tuple: (should_interrupt, interrupt_type)
                interrupt_type: 'MISSILE', 'RWR', 'MAR', None
        """
        try:
            # 检查导弹来袭
            if self.threat_evaluator.detect_incoming_missiles(my_aircraft, env):
                return True, 'MISSILE'

            # 检查RWR告警
            rwr_level = self.threat_evaluator.calculate_rwr_level(my_aircraft, enemy_aircraft_list)
            if rwr_level >= 4:
                return True, 'RWR'

            # 检查MAR距离
            if enemy_aircraft_list:
                for enemy_ac in enemy_aircraft_list:
                    if enemy_ac and enemy_ac.is_alive:
                        my_pos = my_aircraft.get_position()
                        enemy_pos = enemy_ac.get_position()
                        distance = np.linalg.norm(np.array(enemy_pos) - np.array(my_pos))

                        if distance <= 40000:  # MAR = 40km
                            return True, 'MAR'

            return False, None

        except Exception as e:
            logging.error(f"紧急中断检查错误: {e}")
            return False, None

