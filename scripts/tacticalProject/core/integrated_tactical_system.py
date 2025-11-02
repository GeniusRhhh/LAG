"""
集成战术系统
整合态势评估、意图识别、决策制定和机动执行
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional

from .situation_evaluator import SituationEvaluator, SituationScore, TacticalPhase
from .intent_recognizer import IntentRecognizer, EnemyIntent, FriendlyIntent
from .decision_maker import DecisionMaker, ThreatDecision, ThreatResponse
from .maneuver_library import ManeuverLibrary, ManeuverType


class IntegratedTacticalSystem:
    """
    集成战术系统
    
    协调所有子系统，提供统一的决策接口
    """
    
    def __init__(self, my_intent: FriendlyIntent = FriendlyIntent.CONSERVATIVE_CLEAR):
        """
        初始化集成战术系统
        
        Args:
            my_intent: 我方作战意图
        """
        self.my_intent = my_intent
        
        # 初始化子系统
        self.situation_evaluator = SituationEvaluator()
        self.intent_recognizer = IntentRecognizer()
        self.decision_maker = DecisionMaker()
        self.maneuver_library = ManeuverLibrary()
        
        # 数据缓存
        self.situation_cache = {}    # agent_id -> SituationScore
        self.threat_cache = {}       # agent_id -> float
        self.intent_cache = {}       # enemy_id -> EnemyIntent
        self.decision_cache = {}     # agent_id -> ThreatDecision
        
        logging.info(f"✅ 集成战术系统初始化完成 (意图: {my_intent.value})")
    
    def update_situation_assessment(
        self,
        env,
        my_agent_id: str,
        enemy_agent_id: str,
        phase: TacticalPhase
    ):
        """
        更新态势评估
        
        Args:
            env: 环境
            my_agent_id: 我方飞机ID
            enemy_agent_id: 敌方飞机ID
            phase: 当前战术阶段
        """
        try:
            my_aircraft = env.agents[my_agent_id]
            enemy_aircraft = env.agents[enemy_agent_id]
            
            if not my_aircraft.is_alive or not enemy_aircraft.is_alive:
                return
            
            # 评估态势
            situation = self.situation_evaluator.evaluate_situation(
                my_aircraft,
                enemy_aircraft,
                phase
            )
            
            # 计算威胁
            threat = self.situation_evaluator.calculate_threat(situation)
            
            # 缓存结果
            cache_key = f"{my_agent_id}_vs_{enemy_agent_id}"
            self.situation_cache[cache_key] = situation
            self.threat_cache[cache_key] = threat
            
            # 每60步打印一次态势
            if env.current_step % 60 == 0:
                logging.info(
                    f"📊 [{my_agent_id}] 态势评估 (vs {enemy_agent_id}): "
                    f"总分={situation.total:.2f} "
                    f"(角度={situation.angle:.2f} "
                    f"距离={situation.distance:.2f} "
                    f"高度={situation.altitude:.2f} "
                    f"速度={situation.speed:.2f} "
                    f"探测={situation.detection:.2f}) "
                    f"威胁={threat:.2f}"
                )
        
        except Exception as e:
            logging.error(f"态势评估更新失败 {my_agent_id}: {e}")
    
    def update_intent_recognition(
        self,
        env,
        enemy_agent_id: str,
        my_aircraft_list: List
    ):
        """
        更新意图识别
        
        Args:
            env: 环境
            enemy_agent_id: 敌方飞机ID
            my_aircraft_list: 我方飞机列表
        """
        try:
            if enemy_agent_id not in env.agents or not env.agents[enemy_agent_id].is_alive:
                self.intent_cache[enemy_agent_id] = EnemyIntent.ESCAPE
                return
            
            # 识别意图
            intent = self.intent_recognizer.recognize_enemy_intent(
                env,
                enemy_agent_id,
                my_aircraft_list
            )
            
            # 缓存结果
            self.intent_cache[enemy_agent_id] = intent
            
            # 每60步打印一次意图
            if env.current_step % 60 == 0:
                confidence = self.intent_recognizer.get_intent_confidence(enemy_agent_id)
                logging.info(
                    f"🎯 [敌方{enemy_agent_id}] 意图识别: {intent.value} "
                    f"(置信度={confidence:.2f})"
                )
        
        except Exception as e:
            logging.error(f"意图识别更新失败 {enemy_agent_id}: {e}")
    
    def make_tactical_decision(
        self,
        env,
        my_agent_id: str,
        enemy_agent_id: str,
        phase: TacticalPhase
    ) -> ThreatDecision:
        """
        制定战术决策
        
        Args:
            env: 环境
            my_agent_id: 我方飞机ID
            enemy_agent_id: 敌方飞机ID
            phase: 当前战术阶段
            
        Returns:
            ThreatDecision: 威胁决策结果
        """
        try:
            # 获取态势和意图
            cache_key = f"{my_agent_id}_vs_{enemy_agent_id}"
            situation = self.situation_cache.get(cache_key)
            threat = self.threat_cache.get(cache_key, 0.5)
            enemy_intent = self.intent_cache.get(enemy_agent_id, EnemyIntent.ATTACK)
            
            if situation is None:
                # 如果没有态势数据，使用默认值
                from .situation_evaluator import SituationScore
                situation = SituationScore(0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
            
            # 制定决策
            decision = self.decision_maker.make_threat_decision(
                self.my_intent,
                enemy_intent,
                threat,
                situation,
                phase
            )
            
            # 缓存决策
            self.decision_cache[my_agent_id] = decision
            
            # 打印决策
            if env.current_step % 60 == 0 or decision.response != ThreatResponse.CONTINUE:
                logging.info(
                    f"⚖️ [{my_agent_id}] 战术决策: {decision.response.value} "
                    f"({decision.reason})"
                )
            
            return decision
        
        except Exception as e:
            logging.error(f"战术决策失败 {my_agent_id}: {e}")
            # 返回默认决策
            return ThreatDecision(
                response=ThreatResponse.CONTINUE,
                reason="决策失败，默认继续",
                threat_level=0.5
            )
    
    def execute_maneuver(
        self,
        env,
        agent_id: str,
        maneuver_type: ManeuverType,
        **kwargs
    ) -> Tuple[int, int, int]:
        """
        执行机动动作
        
        Args:
            env: 环境
            agent_id: 飞机ID
            maneuver_type: 机动类型
            **kwargs: 机动参数
            
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 指令索引
        """
        try:
            if maneuver_type == ManeuverType.TACTICAL_CLIMB:
                return self.maneuver_library.execute_tactical_climb(env, agent_id, **kwargs)
            elif maneuver_type == ManeuverType.TACTICAL_DESCEND:
                return self.maneuver_library.execute_tactical_descend(env, agent_id, **kwargs)
            elif maneuver_type == ManeuverType.NOTCH_BACK:
                return self.maneuver_library.execute_notch_back(env, agent_id, **kwargs)
            elif maneuver_type == ManeuverType.BEAM:
                return self.maneuver_library.execute_beam_maneuver(env, agent_id, **kwargs)
            elif maneuver_type == ManeuverType.CRANK:
                return self.maneuver_library.execute_crank(env, agent_id, **kwargs)
            elif maneuver_type == ManeuverType.SHORT_SKATE:
                return self.maneuver_library.execute_short_skate(env, agent_id, **kwargs)
            else:
                # 默认平飞
                return 7, 8, 3
        
        except Exception as e:
            logging.error(f"机动执行失败 {agent_id}: {e}")
            return 7, 8, 3
    
    def should_launch_missile(
        self,
        env,
        agent_id: str,
        target_id: str,
        phase: TacticalPhase,
        has_launched: bool
    ) -> bool:
        """
        判断是否应该发射导弹
        
        Args:
            env: 环境
            agent_id: 我方飞机ID
            target_id: 目标ID
            phase: 当前阶段
            has_launched: 是否已发射
            
        Returns:
            bool: 是否应该发射
        """
        try:
            # 获取决策
            decision = self.decision_cache.get(agent_id)
            if decision is None:
                decision = ThreatDecision(
                    response=ThreatResponse.CONTINUE,
                    reason="无决策缓存",
                    threat_level=0.5
                )
            
            # 计算距离
            my_pos = env.agents[agent_id].get_position()
            target_pos = env.agents[target_id].get_position()
            distance = np.linalg.norm(np.array(target_pos) - np.array(my_pos))
            
            # 调用决策器判断
            should_launch = self.decision_maker.should_launch_missile(
                phase,
                distance,
                decision,
                has_launched
            )
            
            if should_launch:
                logging.info(f"🚀 [{agent_id}] 满足导弹发射条件 (阶段={phase.value}, 距离={distance/1000:.1f}km)")
            
            return should_launch
        
        except Exception as e:
            logging.error(f"导弹发射判断失败 {agent_id}: {e}")
            return False
    
    def get_situation(self, my_agent_id: str, enemy_agent_id: str) -> Optional[SituationScore]:
        """获取缓存的态势评估"""
        cache_key = f"{my_agent_id}_vs_{enemy_agent_id}"
        return self.situation_cache.get(cache_key)
    
    def get_threat(self, my_agent_id: str, enemy_agent_id: str) -> float:
        """获取缓存的威胁值"""
        cache_key = f"{my_agent_id}_vs_{enemy_agent_id}"
        return self.threat_cache.get(cache_key, 0.5)
    
    def get_intent(self, enemy_agent_id: str) -> EnemyIntent:
        """获取缓存的敌方意图"""
        return self.intent_cache.get(enemy_agent_id, EnemyIntent.ATTACK)
    
    def get_decision(self, my_agent_id: str) -> Optional[ThreatDecision]:
        """获取缓存的决策"""
        return self.decision_cache.get(my_agent_id)
