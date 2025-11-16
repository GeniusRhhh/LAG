#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
战术系统集成模块（Tactical System Integration Module）

功能：
    1. 集成所有战术子模块
    2. 提供统一的战术系统接口
    3. 协调各模块之间的交互
    4. 管理整体战术流程

作者：TacticalProject重构组
版本：2.0.0
日期：2024-11-13
"""

import logging
import numpy as np
from typing import Dict, Tuple, Optional, Any
from enum import Enum

# 导入所有子模块
from .tactics_executor_full import TacticsExecutor, TacticName
from .tactics_methods import *
from .maneuver_methods import ManeuverMethods
from .node_decision_maker import NodeDecisionMaker
from .node_decision_methods import NodeDecisionMethods
from .phase_manager import PhaseManager, TacticalPhase
from .formation_controller import FormationController, FormationType
from .tactical_selector_algorithm import TacticalSelectorAlgorithm as TacticalSelector
from .decision_manager import TacticalDecisionManager
from .decision_table import DecisionTable


class IntegratedTacticalSystem:
    """
    集成战术系统（Integrated Tactical System）
    
    职责：
        1. 集成和协调所有战术子系统
        2. 提供统一的对外接口
        3. 管理战术执行流程
        4. 处理模块间的数据流
    
    架构设计：
        - 外观模式：为复杂子系统提供简单接口
        - 中介者模式：协调各模块之间的交互
        - 单例模式：确保系统唯一实例
    """
    
    def __init__(self, our_intent: str = 'CONSERVATIVE_CLEAR', 
                 logger: Optional[logging.Logger] = None):
        """
        初始化集成战术系统
        
        Args:
            our_intent: 我方意图
            logger: 日志记录器
        """
        # 日志配置
        self.logger = logger or logging.getLogger(__name__)
        
        # 系统参数
        self.our_intent = our_intent
        self.is_initialized = False
        
        # ========== 初始化所有子模块 ==========
        self.logger.info("🚀 [IntegratedTacticalSystem] 开始初始化集成战术系统...")
        
        # 1. 战术执行器
        self.tactics_executor = self._init_tactics_executor()
        
        # 2. 节点决策器
        self.node_decision_maker = self._init_node_decision_maker()
        
        # 3. 阶段管理器
        self.phase_manager = PhaseManager()
        
        # 4. 编队控制器
        self.formation_controller = FormationController()
        
        # 5. 战术选择器
        # 注：TacticalSelectorAlgorithm需要threat_evaluator和decision_table参数
        # 这里先创建空实例，后续在_setup_module_connections中设置
        self.tactical_selector = None
        
        # 6. 决策管理器
        self.decision_manager = TacticalDecisionManager(our_intent_type=our_intent)
        
        # 7. 决策表
        self.decision_table = DecisionTable()
        
        # ========== 模块间关联 ==========
        self._setup_module_connections()
        
        # ========== 系统状态 ==========
        self.current_tactic = None
        self.current_phase = TacticalPhase.NLT_MELD
        self.agent_phases = {}
        self.is_second_attack = False
        
        self.is_initialized = True
        self.logger.info("✅ [IntegratedTacticalSystem] 集成战术系统初始化完成")
        self.logger.info(f"   我方意图: {our_intent}")
        self.logger.info(f"   支持战术: {[t.value for t in TacticName]}")
    
    def _init_tactics_executor(self) -> TacticsExecutor:
        """
        初始化战术执行器，集成所有执行方法
        
        Returns:
            完整的战术执行器实例
        """
        # 创建基础执行器
        executor = TacticsExecutor(logger=self.logger)
        
        # 动态添加战术执行方法
        import types
        import inspect
        from . import tactics_methods
        
        # 绑定所有execute_*方法
        for method_name in dir(tactics_methods):
            if method_name.startswith('execute_'):
                method = getattr(tactics_methods, method_name)
                # 检查是否是函数（而不是已经绑定的方法）
                if inspect.isfunction(method):
                    setattr(executor, method_name, types.MethodType(method, executor))
        
        # 绑定所有机动方法
        maneuver_methods = ManeuverMethods()
        for method_name in dir(maneuver_methods):
            if method_name.startswith('_') and not method_name.startswith('__'):
                method = getattr(maneuver_methods, method_name)
                # 绑定方法时，需要保证method是未绑定的函数
                bound_method = lambda self, *args, method=method, **kwargs: method(*args, **kwargs)
                setattr(executor, method_name, types.MethodType(bound_method, executor))
        
        self.logger.info("   ✅ 战术执行器初始化完成")
        return executor
    
    def _init_node_decision_maker(self) -> NodeDecisionMaker:
        """
        初始化节点决策器，集成所有决策方法
        
        Returns:
            完整的节点决策器实例
        """
        # 创建决策器（已经继承了 NodeDecisionMethods，所以直接有所有方法）
        decision_maker = NodeDecisionMaker(logger=self.logger)
        
        self.logger.info("   ✅ 节点决策器初始化完成")
        return decision_maker
    
    def _setup_module_connections(self):
        """
        设置模块间的连接和依赖关系
        """
        # 创建战术选择器（需要threat_evaluator和decision_table）
        self.tactical_selector = TacticalSelector(
            threat_evaluator=self.decision_manager.threat_evaluator,
            decision_table=self.decision_table
        )
        
        # 节点决策器需要访问其他组件
        self.node_decision_maker.decision_table = self.decision_table
        self.node_decision_maker.tactical_selector = self.tactical_selector
        self.node_decision_maker.threat_evaluator = self.decision_manager.threat_evaluator
        self.node_decision_maker.situation_evaluator = self.decision_manager.situation_evaluator
        self.node_decision_maker.intent_recognizer = self.decision_manager.enemy_intent_recognizer
        
        # 战术选择器设置
        self.tactical_selector.current_tactic = None
        
        self.logger.info("   ✅ 模块连接设置完成")
    
    # ==================== 公共接口 ====================
    
    def get_tactical_command(self, env, agent_id: str) -> Tuple[int, int, int]:
        """
        获取战术指令（主接口方法）
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
        
        Returns:
            (alt_cmd, hdg_cmd, vel_cmd): 控制指令三元组
        """
        if not self.is_initialized:
            self.logger.error("系统未初始化")
            return 7, 8, 3  # 默认平飞
        
        try:
            # 1. 更新阶段
            phase = self.phase_manager.update_phase(env, agent_id, self.current_tactic)
            self.agent_phases[agent_id] = phase
            
            # 2. 检查是否为决策节点
            if self.phase_manager.is_decision_node(phase):
                # 构建决策上下文
                context = self._build_decision_context(env, agent_id, phase)
                
                # 进行决策
                decision = self.node_decision_maker.make_decision(
                    self._phase_to_node(phase), 
                    context
                )
                
                # 处理决策结果
                self._process_decision(decision)
            
            # 3. 执行当前战术
            if self.current_tactic:
                command = self.tactics_executor.execute_tactic(
                    self.current_tactic,
                    env,
                    agent_id,
                    phase=phase,
                    is_second_attack=self.is_second_attack
                )
            else:
                # 默认直飞
                command = (7, 8, 3)
            
            # 4. 应用编队控制（如果需要）
            if agent_id.endswith('200'):  # 僚机
                formation_params = self.formation_controller.calculate_formation_params(
                    env, 'A0100', agent_id
                )
                if formation_params.get('formation_established', False):
                    # 编队已建立，可能需要调整指令
                    command = self._adjust_command_for_formation(
                        command, formation_params, agent_id, env
                    )
            
            return command
            
        except Exception as e:
            self.logger.error(f"战术系统错误: {e}", exc_info=True)
            return 7, 8, 3  # 默认平飞
    
    def update_tactical_situation(self, env):
        """
        更新战术态势
        
        Args:
            env: 环境对象
        """
        # 更新全局阶段
        self.current_phase = self.phase_manager.update_phase(env)
        
        # 更新决策管理器
        # TacticalDecisionManager没有update_situation方法，这里先注释掉
        # self.decision_manager.update_situation(env)
        
        # 检查第二轮攻击标记
        if self.phase_manager.is_second_attack:
            self.is_second_attack = True
    
    def set_tactical_intent(self, intent: str):
        """
        设置我方战术意图
        
        Args:
            intent: 意图类型
        """
        self.our_intent = intent
        self.decision_manager.our_intent.set_intent(intent)
        self.logger.info(f"📋 战术意图更新: {intent}")
    
    def force_tactic_selection(self, tactic: str):
        """
        强制选择特定战术（用于测试）
        
        Args:
            tactic: 战术名称
        """
        if tactic in [t.value for t in TacticName]:
            self.current_tactic = tactic
            self.tactical_selector.current_tactic = tactic
            self.logger.info(f"🎯 强制选择战术: {tactic}")
        else:
            self.logger.warning(f"无效战术: {tactic}")
    
    # ==================== 内部辅助方法 ====================
    
    def _build_decision_context(self, env, agent_id: str, phase: TacticalPhase) -> Any:
        """
        构建决策上下文
        
        Args:
            env: 环境对象
            agent_id: 智能体ID
            phase: 当前阶段
        
        Returns:
            决策上下文对象
        """
        from .node_decision_maker import NodeContext
        
        # 计算基础信息
        distance = self.phase_manager.calculate_distance(env)
        current_time = env.current_step * env.time_interval
        
        # 获取评估结果
        # 修复威胁评估调用 - ThreatEvaluator没有evaluate_threat方法
        my_aircraft = env.agents.get(agent_id)
        enemy_aircraft = None
        for eid in ['B0100', 'B0200']:
            if eid in env.agents and env.agents[eid].is_alive:
                enemy_aircraft = env.agents[eid]
                break
        
        if my_aircraft and enemy_aircraft:
            threat_level = self.decision_manager.threat_evaluator.calculate_total_threat(
                my_aircraft, enemy_aircraft, env
            )
            # 修复evaluate_situation调用 - 需要传入my_aircraft, enemy_aircraft, phase
            from core.situation_evaluator import TacticalPhase
            situation_score = self.decision_manager.situation_evaluator.evaluate_situation(
                my_aircraft, enemy_aircraft, TacticalPhase.NLT_MELD
            )
            # 将 SituationScore 转换为态势字符串
            if situation_score.total >= 0.6:
                situation = 'ADVANTAGE'
            elif situation_score.total >= 0.4:
                situation = 'NEUTRAL'
            else:
                situation = 'DISADVANTAGE'
        else:
            threat_level = 0.0
            situation = 'NEUTRAL'
        # 修复enemy_intent调用 - 使用实际的飞机状态
        if my_aircraft and enemy_aircraft:
            # 导入Catalog以获取航向
            from envs.JSBSim.core.catalog import Catalog as c
            
            # 获取航向（yaw角，从psi_rad转换为度）
            my_heading = np.rad2deg(my_aircraft.get_property_value(c.attitude_psi_rad))
            enemy_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))
            
            my_state = {
                'position': np.array(my_aircraft.get_position()),
                'velocity': np.array(my_aircraft.get_velocity()),
                'heading': my_heading
            }
            enemy_state = {
                'position': np.array(enemy_aircraft.get_position()),
                'velocity': np.array(enemy_aircraft.get_velocity()),
                'heading': enemy_heading
            }
            enemy_intent = self.decision_manager.enemy_intent_recognizer.recognize(my_state, enemy_state)
        else:
            enemy_intent = 'unknown'
        
        # 统计存活信息
        enemies_alive = sum(1 for eid in ['B0100', 'B0200'] 
                          if eid in env.agents and env.agents[eid].is_alive)
        
        # 获取飞机列表
        my_aircraft_list = [env.agents.get(agent_id)] if env.agents.get(agent_id) else []
        enemy_aircraft_list = [env.agents.get(eid) for eid in ['B0100', 'B0200'] if env.agents.get(eid) and env.agents.get(eid).is_alive]
        
        # 构建上下文
        context = NodeContext(
            env=env,
            agent_id=agent_id,
            current_time=current_time,
            distance=distance,
            threat_level=threat_level,
            situation=situation,
            enemy_intent=enemy_intent,
            our_intent=self.our_intent,
            my_aircraft=my_aircraft_list,
            enemy_aircraft=enemy_aircraft_list,
            is_second_attack=self.is_second_attack,
            custom_data={
                'current_tactic': self.current_tactic,
                'enemies_alive': enemies_alive,
                'phase': phase.value,
                'fuel_remaining': 80,  # 简化：假设80%燃油
                'missiles_remaining': 1,  # 简化：假设1枚导弹
            }
        )
        
        return context
    
    def _process_decision(self, decision: Dict[str, Any]):
        """
        处理决策结果
        
        Args:
            decision: 决策结果字典
        """
        # 更新战术
        if decision.get('tactic'):
            self.current_tactic = decision['tactic']
            self.tactical_selector.current_tactic = decision['tactic']
            self.logger.info(f"🎯 战术更新: {decision['tactic']}")
        
        # 处理编队调整
        if decision.get('formation'):
            self.formation_controller.apply_tactical_formation(
                self.current_tactic,
                decision['formation']
            )
        
        # 处理第二轮攻击标记
        if decision.get('is_second_attack'):
            self.is_second_attack = True
            self.logger.info("🔄 进入第二轮攻击")
        
        # 处理撤退标记
        if decision.get('retreat'):
            self.current_tactic = 'TACTICAL_TURN'
            self.logger.info("🚁 执行撤退")
    
    def _phase_to_node(self, phase: TacticalPhase) -> str:
        """
        将阶段转换为节点名称
        
        Args:
            phase: 战术阶段
        
        Returns:
            节点名称
        """
        phase_node_map = {
            TacticalPhase.NLT_MELD: 'NLT',
            TacticalPhase.MELD_MTR: 'MELD',
            TacticalPhase.MTR_LR: 'MTR',
            TacticalPhase.LR_TR: 'LR',
            TacticalPhase.TR_DOR: 'TR',
            TacticalPhase.DOR_DR: 'DOR',
            TacticalPhase.DR_MAR: 'DR',
            TacticalPhase.BEYOND_MAR: 'MAR',
        }
        
        node = phase_node_map.get(phase, 'NLT')
        
        # 第二轮攻击节点
        if self.is_second_attack:
            if node == 'MTR':
                return 'MTR2'
            elif node == 'LR':
                return 'LR2'
            elif node == 'TR':
                return 'TR2'
        
        return node
    
    def _adjust_command_for_formation(self, command: Tuple[int, int, int], 
                                     formation_params: Dict, 
                                     agent_id: str, env) -> Tuple[int, int, int]:
        """
        根据编队参数调整指令
        
        Args:
            command: 原始指令
            formation_params: 编队参数
            agent_id: 智能体ID
            env: 环境对象
        
        Returns:
            调整后的指令
        """
        # 如果编队误差较大，可能需要调整
        if formation_params.get('total_error', 0) > 1000:
            # 使用编队控制器计算调整
            adjusted = self.formation_controller.calculate_formation_commands(
                env, formation_params, agent_id
            )
            return adjusted
        
        return command
    
    def get_status(self) -> Dict[str, Any]:
        """
        获取系统状态
        
        Returns:
            系统状态字典
        """
        return {
            'initialized': self.is_initialized,
            'current_tactic': self.current_tactic,
            'current_phase': self.current_phase.value if self.current_phase else None,
            'our_intent': self.our_intent,
            'is_second_attack': self.is_second_attack,
            'agent_phases': {k: v.value for k, v in self.agent_phases.items()},
        }
