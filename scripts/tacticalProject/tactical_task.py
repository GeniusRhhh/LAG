"""
通用战术任务类 - 重构版本（模块化架构）
将4706行代码重构为~600行，通过模块化提高可维护性
Version: 2024-11-13 - 模块化重构
"""
import os
import logging
from enum import Enum
import numpy as np
import torch
from typing import Dict, Tuple
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition

# 导入核心模块
from core.threat_evaluator import ThreatEvaluator
from core.situation_evaluator import SituationEvaluator
from core.intent_predictor import IntentPredictor
from core.decision_table import DecisionTable
from core.tactic_selector_v2 import TacticSelectorV2
from core.complete_tactical_system import CompleteTacticalSystem
from core.target_assignment import get_fixed_target
from tactical_types import TacticalPhase, get_target_with_fallback
from enemy_ai_adapter import EnemyAIAdapter
from utils.data_logger import DataLogger
from utils.constants import CONTROL_RANGES

# 导入新重构的模块
from tactical_state_manager import TacticalStateManager
from missile_manager import MissileManager
from tactical_decision_maker import TacticalDecisionMaker
from tactical_executor import TacticalExecutor
from maneuver_library import ManeuverLibrary
from tactical_utils import TacticalUtils
from core.node_decision_maker import NodeDecisionMaker, NodeContext
from simulation.radar_manager import update_all_radars, get_friendly_radar_state, get_enemy_radar_state, get_unified_radar_manager
from core.decision_table import DecisionTable
from core.tactical_selector_algorithm import TacticalSelectorAlgorithm
from core.intent_recognizer import IntentRecognizer

# 导入雷达系统
import sys
radar_path = os.path.join(os.path.dirname(__file__), '..', 'tacticalTemplateProject')
if radar_path not in sys.path:
    sys.path.insert(0, radar_path)
try:
    from radar_manager import UnifiedRadarManager, APG68RadarModel, N001VERadarModel
    RADAR_AVAILABLE = True
except ImportError:
    logging.warning("雷达系统未找到，将使用简化模拟")
    RADAR_AVAILABLE = False


class TacticalTermination(BaseTerminationCondition):
    """战术任务终止条件"""
    
    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)
        self.max_steps = getattr(config, 'max_steps', 1500)
    
    def get_termination(self, task, env, agent_id, info={}):
        """终止条件检查"""
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} 高度过低: {current_alt:.1f}m")
            return True, False, info
        
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} 检测到极端状态")
            return True, False, info
        
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} 过载")
            return True, False, info
        
        # 检查双方存活状态（优先级最高）
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]
        
        if len(red_alive) == 0:
            self.log("红方全灭")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            self.log("蓝方全灭")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}
        
        # 时间限制检查（8分钟=2400步）
        max_time_steps = 2400  # 8分钟 @ 0.2s/步
        max_steps_limit = getattr(env, 'max_steps', max_time_steps)
        
        if env.current_step >= max_steps_limit:
            current_time = env.current_step * getattr(env, 'time_interval', 0.2)
            self.log(f"达到最大时间限制: {current_time:.0f}秒 ({env.current_step} 步)")
            return True, False, {"termination_reason": "time_limit", "winner": "draw"}
        
        return False, False, info


class TacticalTask(MultipleCombatTask):
    """
    通用战术任务类（重构版）
    通过模块化架构将原4706行代码精简为~600行
    """
    
    def __init__(self, config, decision_manager=None, force_tactic=None):
        super().__init__(config)

        self.termination_conditions = [TacticalTermination(self.config)]

        # 🎯 强制战术选择配置
        # 处理字符串 'None' 转换为 Python 对象 None
        if force_tactic == 'None':
            self.force_tactic = None
        else:
            self.force_tactic = force_tactic
        
        # ===== 初始化模块化组件 =====
        # 1. 状态管理器
        self.state_manager = TacticalStateManager()
        
        # 2. 导弹管理器
        self.missile_manager = MissileManager(self.state_manager)
        
        # 3. 返航状态跟踪
        self.returning_agents = set()  # 正在返航的智能体
        
        # 4. 集成战术系统
        from core.integrated_tactical_system import IntegratedTacticalSystem
        from core.intent_recognizer import FriendlyIntent
        
        my_intent_str = getattr(config, 'friendly_intent', 'CONSERVATIVE_CLEAR')
        if my_intent_str == 'AGGRESSIVE_CLEAR':
            my_intent = FriendlyIntent.AGGRESSIVE_CLEAR
        elif my_intent_str == 'DEFENSIVE':
            my_intent = FriendlyIntent.DEFENSIVE
        else:
            my_intent = FriendlyIntent.CONSERVATIVE_CLEAR
        
        self.integrated_system = IntegratedTacticalSystem(my_intent=my_intent)
        self.complete_tactical_system = CompleteTacticalSystem(my_intent=my_intent_str)
        logging.info(f"✅ 集成战术系统已启动 (我方意图: {my_intent.value})")
        self.my_intent_str = my_intent_str
        
        # 4. 决策制定器（需要威胁和态势评估器）
        self.threat_evaluator = ThreatEvaluator()
        self.situation_evaluator = SituationEvaluator()
        self.decision_maker = TacticalDecisionMaker(
            self.state_manager,
            self.threat_evaluator,
            self.situation_evaluator
        )
        
        # 5. 战术执行器
        self.executor = TacticalExecutor(self)
        
        # 6. 机动库
        self.maneuver_lib = ManeuverLibrary(self)
        self.node_decision_maker = NodeDecisionMaker()
        # 绑定节点决策器依赖
        try:
            # 使用完整系统中的评估器与选择器，避免方法缺失（detect_incoming_missiles等）
            self.node_decision_maker.decision_table = self.complete_tactical_system.decision_table
            self.node_decision_maker.tactical_selector = self.complete_tactical_system.tactical_selector
            self.node_decision_maker.threat_evaluator = self.complete_tactical_system.threat_evaluator
            self.node_decision_maker.situation_evaluator = self.situation_evaluator
            self.node_decision_maker.intent_recognizer = IntentRecognizer()
        except Exception as e:
            logging.warning(f"节点决策器依赖绑定失败: {e}")
        
        # ===== 保留的原有组件 =====
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False, "phase": None, "start_time": 0,
            "completed": False, "completed_time": 0
        }
        self.is_second_attack = False
        self.attack_waypoint = {}
        
        self.TARGET_LONGITUDINAL_MIN = 5000
        self.TARGET_LONGITUDINAL_MAX = 11000
        self.TARGET_LATERAL_MAX = 500
        
        self.initial_heading = {}
        self.initial_altitude = {}
        self._inner_rnn_states = {}
        
        # 数据记录器
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_root_dir() + '/scripts/tacticalProject/tactical_simulation_results'
        self.data_logger = DataLogger(output_dir, timestamp)
        logging.info("✅ 数据记录器初始化完成")
        
        # 雷达系统
        if RADAR_AVAILABLE:
            try:
                self.radar_manager = UnifiedRadarManager()
                logging.info("✅ 雷达系统初始化完成 (APG-68 vs N001VE)")
            except Exception as e:
                logging.warning(f"雷达系统初始化失败: {e}")
                self.radar_manager = None
        else:
            self.radar_manager = None
        
        # Baseline模型
        self._load_baseline_model()
        
        # CAP区域约束
        self.cap_boundary = {
            'type': 'racetrack', 'center_x': 0, 'center_y': 0,
            'length': 240000, 'width': 120000, 'z_min': 3000, 'z_max': 18000,
            'rect_x_min': -60000, 'rect_x_max': 60000,
            'rect_y_min': -60000, 'rect_y_max': 60000,
            'circle_radius': 60000,
            'south_circle_center': (-60000, 0),
            'north_circle_center': (60000, 0),
        }
        logging.info(f"✅ CAP区域设置: 南北向跑马道（已扩大）")
        logging.info(f"   长轴: 240km (南北/X方向)")
        logging.info(f"   短轴: 120km (东西/Y方向)")
        
        # 战术模板核心数据结构
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0
        
        self.norm_delta_heading = np.array([
            -np.pi, -2*np.pi/3, -np.pi/2, -5*np.pi/12, -np.pi/3, -np.pi/4, -np.pi/6, -np.pi/12,
            0, np.pi/12, np.pi/6, np.pi/4, np.pi/3, 5*np.pi/12, np.pi/2, 2*np.pi/3, np.pi
        ])
        
        self.norm_delta_velocity = np.array([-150, -100, -50, 0, 50, 100, 150]) / 100.0
        
        # 战术距离节点
        self.tactical_distances = getattr(config, 'tactical_distances', {
            'NLT': 120000, 'MELD': 100000, 'MTR': 80000,
            'LR': 78000, 'TR': 75000, 'DOR': 70000,
            'DR': 65000, 'MAR': 40000
        })
        
        # 战术选择相关
        self.selected_tactic = None
        self.tactic_roles = None
        self.decision_made = {'NLT': False, 'MELD': False, 'MTR': False}
        self.lr_maneuver = {}
        self.lr_launch_count = {}
        self.lr_maneuver = {}  # LR阶段机动决策（'crank' 或 'straight'）
        self.lr_launch_count = {}  # LR发射次数统计
        
        # 敌方AI
        self.enemy_ai = EnemyAIAdapter(project_name="tactical_simulation")
        logging.info("✅ 敌方AI系统初始化完成")
        
        # 导弹系统
        self.missile_system_enabled = True
        logging.info("✅ 导弹系统初始化完成")
        
        # ===== StateManager字典通过@property访问器提供（见文件末尾）=====
        # 不需要在此处赋值，因为已有@property装饰器提供访问
        
        # 日志节流
        self._last_launch_request_log_time = 0.0
        self._last_launch_request_log_step = -1
        
        logging.info("TacticalTask初始化完成")
    
    def _load_baseline_model(self):
        """加载baseline模型"""
        model_path = get_root_dir() + '/model/baseline_model.pt'
        if os.path.exists(model_path):
            self.baseline_model = BaselineActor(input_dim=12, use_mlp_actlayer=False)
            self.baseline_model.load_state_dict(torch.load(model_path, map_location='cpu'))
            self.baseline_model.eval()
            logging.info(f"✅ Baseline模型加载成功: {model_path}")
        else:
            self.baseline_model = None
            logging.warning(f"⚠️ Baseline模型未找到: {model_path}")
    
    def reset(self, env):
        """重置任务状态"""
        # 重置状态管理器
        self.state_manager.reset_all()
        
        # 重置导弹管理器
        self.missile_manager.reset()
        
        # 重置其他状态
        # 🎯 保留强制战术设置，不被重置覆盖
        if not self.force_tactic:
            self.selected_tactic = None
        else:
            # 强制战术模式下，保持强制战术不变
            logging.info(f"🎯 重置时保持强制战术: {self.force_tactic}")
            self.selected_tactic = self.force_tactic
        self.tactic_roles = {}
        self.decision_made = {'NLT': False, 'MELD': False, 'MTR': False}
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False, "phase": None, "start_time": 0,
            "completed": False, "completed_time": 0
        }
        
        # 初始化每架飞机的状态
        for agent_id in env.agents.keys():
            if env.agents[agent_id].is_alive:
                self.state_manager.init_agent_states(agent_id, max_missiles=2)
                self.initial_heading[agent_id] = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
                self.initial_altitude[agent_id] = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                self._inner_rnn_states[agent_id] = torch.zeros(1, 1, 128)
                logging.info(f"{agent_id} 初始状态: 航向{np.rad2deg(self.initial_heading[agent_id]):.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")
        
        # 重置雷达系统（如果有reset方法的话）
        if self.radar_manager and hasattr(self.radar_manager, 'reset'):
            self.radar_manager.reset()
        if hasattr(self, 'node_decision_maker') and hasattr(self.node_decision_maker, 'dr_window_start'):
            self.node_decision_maker.dr_window_start = None
        # 清理占位点
        self.attack_waypoint = {}
        
        # 重置数据记录器（如果有reset方法的话）
        if hasattr(self.data_logger, 'reset'):
            self.data_logger.reset()
        
        logging.info("TacticalTask重置完成")
    
    def get_action(self, env, agent_id):
        """
        主决策入口 - 重构版本
        委托给各个模块处理，保持简洁
        """
        current_time = env.current_step * env.time_interval
        
        # 检查飞机是否存活
        if not env.agents[agent_id].is_alive:
            return self._get_default_action()
        
        # 更新阶段
        self._update_phase(env, agent_id, current_time)
        
        # 更新导弹状态
        self.missile_manager.update_missile_status(env, current_time)
        
        # 处理导弹发射请求（按原版逻辑）
        self._handle_missile_launches(env, current_time)
        
        # 调试信息：输出导弹状态（降低频率）
        if env.current_step % 900 == 0:  # 每180秒输出一次
            for aid in ['A0100', 'A0200']:
                if aid in self.state_manager.missile_launched:
                    logging.info(f"🚀 [{aid}] 导弹状态: launched={self.state_manager.missile_launched[aid]}, 剩余={env.agents[aid].num_missiles}枚")
        
        # 敌方AI决策
        if agent_id.startswith('B'):
            return self._get_enemy_action(env, agent_id)
        
        # 我方决策：根据战术执行
        if self.selected_tactic is None:
            self._select_tactic_at_phase(env, agent_id)
            
        # 调试信息：输出战术选择状态
        if env.current_step % 60 == 0 and agent_id == 'A0100':  # 每12秒输出一次，仅A0100
            logging.info(f"🎯 [{agent_id}] 战术状态: tactic={self.selected_tactic}, phase={self.state_manager.get_agent_phase(agent_id)}")
        
        # 委托给战术执行器
        if env.current_step % 60 == 0 and agent_id == 'A0100':
            logging.info(f"🎯 [{agent_id}] get_action: selected_tactic={self.selected_tactic}")
        
        # 🔥 新增：检查是否在智能重新交战模式
        if self._check_intelligent_reengagement_active(agent_id):
            return self._execute_intelligent_reengagement(env, agent_id, current_time)
        
        if self.selected_tactic == 'DRAG_SHOOT':
            if env.current_step % 60 == 0 and agent_id == 'A0100':
                logging.info(f"🎯 [{agent_id}] 执行DRAG_SHOOT战术")
            return self.executor.execute_drag_shoot(env, agent_id)
        elif self.selected_tactic == 'PINCER_ATTACK':
            return self.executor.execute_pincer_attack(env, agent_id)
        elif self.selected_tactic == 'HIGH_LOW_ATTACK':
            return self.executor.execute_high_low_attack(env, agent_id)
        elif self.selected_tactic == 'FRONT_BACK':
            return self.executor.execute_front_back(env, agent_id)
        elif self.selected_tactic == 'SIDE_BY_SIDE':
            return self.executor.execute_side_by_side(env, agent_id)
        elif self.selected_tactic == 'TACTICAL_EVASION':
            return self.executor.execute_tactical_evasion(env, agent_id)
        elif self.selected_tactic == 'TACTICAL_TURN':
            return self.executor.execute_tactical_turn(env, agent_id)
        elif self.selected_tactic == 'INTELLIGENT_REENGAGEMENT':
            return self._execute_intelligent_reengagement(env, agent_id, current_time)
        elif self.selected_tactic == 'FORMATION_RESET':
            return self._execute_formation_reset_procedure(env, agent_id)
        elif self.selected_tactic == 'UNIFIED_SECOND_ATTACK':
            return self._execute_unified_second_attack(env, agent_id)
        else:
            # 默认：平稳飞行
            action = 7, 8, 3
            return action
        
        # 确保返回值有效
        if action is None:
            logging.warning(f"⚠️ {agent_id} 战术执行返回None，使用默认动作")
            return 7, 8, 4
        elif not isinstance(action, tuple) or len(action) != 3:
            logging.warning(f"⚠️ {agent_id} 战术执行返回格式错误: {action}，使用默认动作")
            return 7, 8, 4
        
        return action
    
    def _update_phase(self, env, agent_id: str, current_time: float):
        """更新飞机所处的战术阶段（原版逻辑：僚机滞后）"""
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            return
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return
        
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        adjusted_distance = distance  # 初始化为实际距离
        
        # ✅ 原版逻辑：僚机滞后，避免频繁震荡
        # 拖曳射击、前后攻击、并排射击都需要僚机滞后
        is_lead = agent_id.endswith('100')
        if not is_lead and self.selected_tactic in ['DRAG_SHOOT', 'FRONT_BACK', 'SIDE_BY_SIDE']:
            # 并排射击：僚机在TR-DOR和DOR-DR阶段滞后
            if self.selected_tactic == 'SIDE_BY_SIDE':
                # 僚机在返航阶段滞后8km（确保长机先返航）
                adjusted_distance += 8000
            else:
                adjusted_distance += 4000  # 其他战术固定滞后4km
        
        # 根据调整后的距离判断阶段
        if adjusted_distance >= self.tactical_distances['NLT']:
            new_phase = TacticalPhase.NLT_MELD
        elif adjusted_distance >= self.tactical_distances['MELD']:
            new_phase = TacticalPhase.MELD_MTR
        elif adjusted_distance >= self.tactical_distances['MTR']:
            new_phase = TacticalPhase.MTR_LR
        elif adjusted_distance >= self.tactical_distances['LR']:
            new_phase = TacticalPhase.LR_TR
        elif adjusted_distance >= self.tactical_distances['TR']:
            new_phase = TacticalPhase.TR_DOR
        elif adjusted_distance >= self.tactical_distances['DOR']:
            new_phase = TacticalPhase.DOR_DR
        elif adjusted_distance >= self.tactical_distances['DR']:
            new_phase = TacticalPhase.DR_MAR
        else:
            new_phase = TacticalPhase.BEYOND_MAR
        
        old_phase = self.state_manager.get_agent_phase(agent_id)
        self.state_manager.set_agent_phase(agent_id, new_phase, current_time)
        
        # ✅ 原版逻辑：长机阶段作为全局阶段
        if is_lead:
            old_global_phase = self.state_manager.current_phase
            if new_phase != old_global_phase:
                self.state_manager.current_phase = new_phase
        
        # 阶段切换时执行决策
        if old_phase != new_phase:
            if new_phase == TacticalPhase.LR_TR:
                self._decide_at_lr(env, agent_id)
                # 基于节点方法的参数/发射评估
                self._node_decision_at_node(env, agent_id, 'LR', current_time)
                if getattr(self, 'is_second_attack', False):
                    self._node_decision_at_node(env, agent_id, 'LR2', current_time)
            elif new_phase == TacticalPhase.MTR_LR:
                # 首轮MTR：占位点/轻微队形调整
                self._node_decision_at_node(env, agent_id, 'MTR', current_time)
                if getattr(self, 'is_second_attack', False):
                    self._node_decision_at_node(env, agent_id, 'MTR2', current_time)
            elif new_phase == TacticalPhase.MELD_MTR:
                # MELD节点：允许战术切换与编队调整（二级控制距离）
                self._node_decision_at_node(env, agent_id, 'MELD', current_time)
            elif new_phase == TacticalPhase.NLT_MELD:
                # NLT节点：策略与战术初始选择（一级控制距离）
                self._node_decision_at_node(env, agent_id, 'NLT', current_time)
            elif new_phase == TacticalPhase.DOR_DR:
                self._node_decision_at_node(env, agent_id, 'DOR', current_time)
            elif new_phase == TacticalPhase.DR_MAR:
                self._node_decision_at_node(env, agent_id, 'DR', current_time)
            elif new_phase == TacticalPhase.BEYOND_MAR:
                self._node_decision_at_node(env, agent_id, 'MAR', current_time)
            elif new_phase == TacticalPhase.TR_DOR:
                self._node_decision_at_node(env, agent_id, 'TR', current_time)
                if getattr(self, 'is_second_attack', False):
                    self._node_decision_at_node(env, agent_id, 'TR2', current_time)
    
    def _select_tactic_at_phase(self, env, agent_id: str):
        """在关键节点选择战术"""
        current_phase = self.state_manager.get_agent_phase(agent_id)
        current_time = env.current_step * env.time_interval
        
        # 战术选择条件：NLT节点 或 MELD节点 或 MTR_LR节点且还未选择战术
        should_select_tactic = False
        phase_name = ""
        
        if current_phase == TacticalPhase.NLT_MELD and not self.decision_made['NLT']:
            should_select_tactic = True
            phase_name = "NLT节点"
            
        elif current_phase == TacticalPhase.MELD_MTR and not self.decision_made['MELD']:
            should_select_tactic = True
            phase_name = "MELD节点"
            
        elif current_phase == TacticalPhase.MTR_LR and self.selected_tactic is None:
            should_select_tactic = True
            phase_name = "MTR_LR节点（强制选择）"
            
        if should_select_tactic:
            # 🎯 优先检查force_tactic参数（来自run_simulation.py）
            if self.force_tactic and self.force_tactic in ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'FRONT_BACK', 'SIDE_BY_SIDE']:
                logging.info(f"🎯 强制选择战术: {self.force_tactic}")
                self.selected_tactic = self.force_tactic
                # 设置默认角色
                if self.force_tactic == 'PINCER_ATTACK':
                    self.tactic_roles = {'lead': 'A0100', 'wingman': 'A0200', 'lead_target': 'B0100', 'wingman_target': 'B0200', 'tactic': 'PINCER_ATTACK'}
                elif self.force_tactic == 'HIGH_LOW_ATTACK':
                    self.tactic_roles = {'high': 'A0100', 'low': 'A0200', 'high_target': 'B0100', 'low_target': 'B0200', 'tactic': 'HIGH_LOW_ATTACK'}
                elif self.force_tactic == 'FRONT_BACK':
                    self.tactic_roles = {'leader': 'A0100', 'wingman': 'A0200', 'leader_target': 'B0100', 'wingman_target': 'B0200', 'tactic': 'FRONT_BACK'}
                elif self.force_tactic == 'TACTICAL_TURN':
                    self.tactic_roles = {'lead': 'A0100', 'wingman': 'A0200', 'lead_target': 'B0100', 'wingman_target': 'B0200', 'tactic': 'TACTICAL_TURN'}
                else:  # DRAG_SHOOT
                    self.tactic_roles = {'lead': 'A0100', 'wingman': 'A0200', 'lead_target': 'B0100', 'wingman_target': 'B0200', 'tactic': 'DRAG_SHOOT'}
            # 检查是否有环境变量强制指定战术（保留向后兼容性）
            else:
                import os
                force_tactic = os.environ.get('FORCE_TACTIC')
                if force_tactic and force_tactic in ['DRAG_SHOOT', 'PINCER_ATTACK', 'HIGH_LOW_ATTACK', 'SEQUENTIAL_ATTACK', 'SIDE_BY_SIDE', 'FRONT_BACK']:
                    logging.info(f"🧪 环境变量强制选择战术: {force_tactic}")
                    self.selected_tactic = force_tactic
                    # 设置默认角色
                    if force_tactic == 'PINCER_ATTACK':
                        self.tactic_roles = {'A0100': 'left', 'A0200': 'right'}
                    elif force_tactic == 'HIGH_LOW_ATTACK':
                        self.tactic_roles = {'A0100': 'high', 'A0200': 'low'}
                    elif force_tactic == 'SEQUENTIAL_ATTACK':
                        self.tactic_roles = {'A0100': 'first', 'A0200': 'second'}
                    elif force_tactic == 'SIDE_BY_SIDE':
                        self.tactic_roles = {'A0100': 'left', 'A0200': 'right'}
                    elif force_tactic == 'FRONT_BACK':
                        self.tactic_roles = {'A0100': 'leader', 'A0200': 'wingman'}
                    else:  # DRAG_SHOOT
                        self.tactic_roles = {'A0100': 'lead', 'A0200': 'wingman'}
                else:
                    # 正常战术选择流程
                    logging.info("=" * 80)
                    logging.info(f"🎯 [{phase_name}] 开始战术决策（完整智能系统）")
                    # 准备飞机列表
                    my_aircraft_list = [env.agents['A0100'], env.agents['A0200']]
                    enemy_aircraft_list = [env.agents['B0100'], env.agents['B0200']]

                    # 调用完整战术系统选择战术（修正参数顺序：control_distance, my_list, enemy_list, env）
                    if current_phase == TacticalPhase.NLT_MELD:
                        control_distance = 'NLT'
                    elif current_phase == TacticalPhase.MELD_MTR:
                        control_distance = 'MELD'
                    else:
                        control_distance = 'MTR'
                    tactic_result = self.complete_tactical_system.select_tactic(
                        control_distance,
                        my_aircraft_list,
                        enemy_aircraft_list,
                        env
                    )

                    # 处理返回结果
                    if isinstance(tactic_result, dict):
                        self.selected_tactic = tactic_result['tactic']
                        self.tactic_roles = tactic_result['roles']
                    else:
                        # 假设返回的是(tactic, roles)元组
                        self.selected_tactic = tactic_result[0]
                        self.tactic_roles = tactic_result[1]

                    logging.info(f"✅ 智能选择战术: {self.selected_tactic}")
                    logging.info(f"✅ 角色分配: {self.tactic_roles}")

            # 更新决策标记
            if current_phase == TacticalPhase.NLT_MELD:
                self.decision_made['NLT'] = True
            elif current_phase == TacticalPhase.MELD_MTR:
                self.decision_made['MELD'] = True

            logging.info(f"   ✅ 最终选定战术: {self.selected_tactic}")
            logging.info(f"   ✅ 最终角色分配: {self.tactic_roles}")
            logging.info("=" * 80)
    
    def _decide_at_lr(self, env, agent_id: str):
        """LR节点决策：根据态势决定是Crank还是平飞"""
        try:
            # 🎯 HIGH_LOW_ATTACK战术特殊处理：僚机不执行Crank，保持高空直飞
            if (self.selected_tactic == 'HIGH_LOW_ATTACK' and 
                self.tactic_roles.get(agent_id) == 'low' and 
                agent_id == 'A0200'):
                self.lr_maneuver[agent_id] = 'straight'
                logging.info(f"🎯 [HIGH_LOW_ATTACK-僚机LR] {agent_id} 战术特定：跳过Crank决策，保持直飞")
                return
            
            # 获取敌机方位
            target_id = get_target_with_fallback(agent_id, env)
            if target_id is None:
                self.lr_maneuver[agent_id] = 'straight'
                return
            
            target_aircraft = env._jsbsims.get(target_id)
            if not target_aircraft or not target_aircraft.is_alive:
                self.lr_maneuver[agent_id] = 'straight'
                return
            
            # 获取当前航向和敌机方位
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 计算敌机方位角（使用原版逻辑）
            my_pos = np.array(env.agents[agent_id].get_position())
            enemy_pos = np.array(target_aircraft.get_position())
            delta_x = enemy_pos[0] - my_pos[0]
            delta_y = enemy_pos[1] - my_pos[1]
            bearing_rad = np.arctan2(delta_y, delta_x)
            bearing_deg = np.rad2deg(bearing_rad)
            enemy_bearing = (90 - bearing_deg) % 360.0  # 从数学坐标系转为航空坐标系
            
            # 计算航向差（带符号，原版公式）
            heading_diff = ((enemy_bearing - current_heading + 180) % 360) - 180
            heading_diff_abs = abs(heading_diff)
            
            # 决策逻辑：航向差大于15度时执行crank（原版逻辑）
            if heading_diff_abs > 15.0:
                self.lr_maneuver[agent_id] = 'crank'
                logging.info(f"🎯 [LR决策] {agent_id} 航向差{heading_diff_abs:.1f}°，选择Crank")
            else:
                self.lr_maneuver[agent_id] = 'straight'
                logging.info(f"📍 [LR决策] {agent_id} 航向差{heading_diff_abs:.1f}°，选择平飞")
                
        except Exception as e:
            logging.error(f"LR决策错误: {e}")
            self.lr_maneuver[agent_id] = 'straight'
    
    def _node_decision_at_node(self, env, agent_id: str, node: str, current_time: float):
        try:
            target_id = get_target_with_fallback(agent_id, env)
            if not target_id:
                return
            target_aircraft = env._jsbsims.get(target_id)
            if not target_aircraft or not target_aircraft.is_alive:
                return
            my_aircraft = env.agents.get(agent_id)
            distance = TacticalUtils.calculate_distance_between(my_aircraft, target_aircraft)
            threat_level = self.threat_evaluator.calculate_total_threat(my_aircraft, target_aircraft, env)
            from core.situation_evaluator import TacticalPhase as SEPhase
            situation_score = self.situation_evaluator.evaluate_situation(my_aircraft, target_aircraft, SEPhase.NLT_MELD)
            if situation_score.total >= 0.6:
                situation = 'ADVANTAGE'
            elif situation_score.total >= 0.4:
                situation = 'NEUTRAL'
            else:
                situation = 'DISADVANTAGE'
            my_aircraft_list = [env.agents.get('A0100'), env.agents.get('A0200')]
            enemy_aircraft_list = [env.agents.get('B0100'), env.agents.get('B0200')]
            enemies_alive = sum(1 for eid in ['B0100', 'B0200'] if eid in env.agents and env.agents[eid].is_alive)
            missiles_remaining = getattr(my_aircraft, 'num_missiles', 0)
            context = NodeContext(
                env=env,
                agent_id=agent_id,
                current_time=current_time,
                distance=distance,
                threat_level=threat_level,
                situation=situation,
                enemy_intent='unknown',
                our_intent=self.my_intent_str,
                my_aircraft=my_aircraft_list,
                enemy_aircraft=enemy_aircraft_list,
                is_second_attack=self.is_second_attack,
                custom_data={
                    'current_tactic': self.selected_tactic,
                    'enemies_alive': enemies_alive,
                    'fuel_remaining': 80,
                    'missiles_remaining': missiles_remaining,
                }
            )
            decision = self.node_decision_maker.make_decision(node, context)
            
            # 🔥 新增：检查是否是DR节点的重新交战决策
            if node == 'DR' and decision.get('reengage', False):
                # **修改：使用新的队形重置和二次进攻系统**
                logging.info(f"⚡ [DR决策] 时间窗口结束，评估是否重新进攻")
                logging.info(f"   → 决定重新进攻！敌机{enemies_alive}架，威胁{threat_level:.2f}")
                
                # 启动队形重置程序
                self._start_formation_reset_procedure(env, agent_id, current_time)
                
                # 切换到队形重置模式
                self.selected_tactic = 'FORMATION_RESET'
                return
            
            if decision.get('tactic'):
                new_tactic = decision['tactic']

                # 🎯 检查是否有强制战术设置，防御战术除外
                if self.force_tactic and new_tactic not in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                    logging.info(f"🎯 节点决策尝试切换到{new_tactic}，但强制战术{self.force_tactic}优先级更高，忽略切换")
                    return  # 忽略非防御战术的切换

                # 若切换到防御类战术，记录之前的战术用于TTL后恢复
                if new_tactic in ['TACTICAL_EVASION', 'TACTICAL_TURN']:
                    if getattr(self, '_defense_prev_tactic', None) is None:
                        # 如果有强制战术，备份强制战术而不是当前战术
                        self._defense_prev_tactic = self.force_tactic if self.force_tactic else self.selected_tactic
                    self.selected_tactic = new_tactic
                    logging.info(f"🛡️ 切换到防御战术{new_tactic}，备份战术: {self._defense_prev_tactic}")
                else:
                    self.selected_tactic = new_tactic
                    # 切回普通战术后清理备份
                    if hasattr(self, '_defense_prev_tactic'):
                        self._defense_prev_tactic = None
            if decision.get('is_second_attack'):
                self.is_second_attack = True
            if decision.get('retreat'):
                self.selected_tactic = 'TACTICAL_TURN'
            if decision.get('attack_waypoint') is not None:
                self.attack_waypoint[agent_id] = decision['attack_waypoint']
        except Exception as e:
            logging.error(f"节点决策错误[{node}]: {e}")
    
    def _get_enemy_action(self, env, agent_id: str):
        """获取敌方行动（使用敌方AI）"""
        current_time = env.current_step * env.time_interval
        # 触发敌方导弹发射逻辑（包含雷达/朝向/MAR检查）
        try:
            if hasattr(self.enemy_ai, 'unified_ai'):
                self.enemy_ai.unified_ai.handle_missile_launch(env, agent_id, current_time)
        except Exception:
            pass
        return self.enemy_ai.get_enemy_command_indices(env, agent_id, current_time)
    
    def _get_default_action(self):
        """默认行动：平稳飞行"""
        return 7, 8, 3
    
    # ===== 辅助方法（供战术执行器和机动库使用） =====
    def _normalize_angle_diff(self, angle_diff: float) -> float:
        """角度差标准化 - 委托给工具类"""
        return TacticalUtils.normalize_angle_diff(angle_diff)
    
    def _calculate_distance_between(self, aircraft1, aircraft2) -> float:
        """计算距离 - 委托给工具类"""
        return TacticalUtils.calculate_distance_between(aircraft1, aircraft2)
    
    def _convert_heading_to_index(self, heading_diff: float) -> int:
        """航向转指令 - 委托给工具类"""
        return TacticalUtils.convert_heading_to_index(heading_diff)
    
    def _convert_altitude_to_index(self, altitude_cmd_value: float) -> int:
        """高度转指令 - 委托给工具类"""
        return TacticalUtils.convert_altitude_to_index(altitude_cmd_value)
    
    def _convert_velocity_to_index(self, velocity_cmd_value: float) -> int:
        """速度转指令 - 委托给工具类"""
        return TacticalUtils.convert_velocity_to_index(velocity_cmd_value)
    
    def _turn_to_heading(self, env, agent_id: str, target_heading: float, speed_cmd=3, altitude_cmd=7) -> tuple:
        """转向到目标航向 - 委托给工具类"""
        return TacticalUtils.turn_to_heading(env, agent_id, target_heading, speed_cmd, altitude_cmd)
    
    def _get_enemy_bearing(self, env, agent_id: str) -> float:
        """获取敌机方位 - 委托给工具类"""
        return TacticalUtils.get_enemy_bearing(env, agent_id)
    
    def _check_cap_boundary(self, pos: tuple) -> bool:
        """检查CAP边界 - 委托给工具类"""
        return TacticalUtils.check_cap_boundary(
            pos, 
            (self.cap_boundary['center_x'], self.cap_boundary['center_y']),
            self.cap_boundary['width'],
            self.cap_boundary['length']
        )
    
    def _get_wingman_phase_by_distance(self, distance: float) -> TacticalPhase:
        """
        根据距离判断僚机阶段（原版逻辑：僚机滞后）
        拖曳射击/前后攻击/并排射击需要僚机滞后，避免频繁震荡
        """
        # ✅ 原版逻辑：僚机滞后设置（从原战术模板）
        wingman_delay_tr_dor = 4000   # TR_DOR阶段滞后4km
        wingman_delay_dor_dr = 8000   # DOR_DR阶段滞后8km
        
        # 根据滞后距离判断阶段
        if distance > self.tactical_distances.get('MELD', 100000):
            return TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances.get('MTR', 80000):
            return TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances.get('LR', 78000):
            return TacticalPhase.MTR_LR
        elif distance > self.tactical_distances.get('TR', 75000):
            return TacticalPhase.LR_TR
        elif distance > (self.tactical_distances.get('DOR', 70000) - wingman_delay_tr_dor):
            return TacticalPhase.TR_DOR  # 70km - 4km = 66km
        elif distance > (self.tactical_distances.get('DR', 65000) - wingman_delay_dor_dr):
            return TacticalPhase.DOR_DR  # 65km - 8km = 57km
        elif distance > self.tactical_distances.get('MAR', 40000):
            return TacticalPhase.DR_MAR
        else:
            return TacticalPhase.BEYOND_MAR
    
    # ===== 机动库方法委托 =====
    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float, duration=10.0) -> tuple:
        """保持航向 - 委托给机动库"""
        return self.maneuver_lib.maintain_heading_precise(env, agent_id, target_heading, duration)
    
    def _execute_short_skate_precise(self, env, agent_id: str, current_time: float, skate_direction: str = None) -> tuple:
        """执行Short Skate机动 - 委托给机动库"""
        return self.maneuver_lib.execute_short_skate_precise(env, agent_id, current_time, skate_direction)
    
    def _execute_tactical_crank(self, env, agent_id: str, direction: str, climb: bool = False) -> tuple:
        """执行战术Crank机动 - 委托给机动库"""
        return self.maneuver_lib.execute_tactical_crank(env, agent_id, direction, climb)
    
    def _establish_rear_formation(self, env, agent_id: str, current_time: float) -> tuple:
        """建立后方编队 - 委托给机动库"""
        return self.maneuver_lib.establish_rear_formation(env, agent_id, current_time)
    
    def _maintain_rear_formation(self, env, agent_id: str) -> tuple:
        """保持后方编队 - 委托给机动库"""
        return self.maneuver_lib.maintain_rear_formation(env, agent_id)
    
    def step(self, env):
        """执行一步仿真 - 核心方法"""
        current_time = env.current_step * env.time_interval
        
        # 更新导弹状态
        self.missile_manager.update_missile_status(env, current_time)
        update_all_radars(env, current_time)
        # 将雷达状态写回飞机以驱动RWR/威胁评估
        try:
            for aid in ['A0100', 'A0200']:
                if aid in env.agents and env.agents[aid].is_alive:
                    mode = get_friendly_radar_state(aid)
                    env.agents[aid].radar_mode = mode.value if hasattr(mode, 'value') else str(mode)
            for bid in ['B0100', 'B0200']:
                if bid in env.agents and env.agents[bid].is_alive:
                    mode = get_enemy_radar_state(bid)
                    env.agents[bid].radar_mode = mode.value if hasattr(mode, 'value') else str(mode)
        except Exception as e:
            if env.current_step % 120 == 0:
                logging.warning(f"雷达状态回写失败: {e}")
        
        # RWR/ECM摘要（每60秒一次）
        if env.current_step % 300 == 0:
            try:
                rm = get_unified_radar_manager()
                for aid in ['A0100', 'A0200']:
                    lvl = rm.get_rwr_threat_level(aid)
                    ecm_on = rm.ecm_states.get(aid, {}).get('active', False)
                    ecm_type = rm.ecm_states.get(aid, {}).get('type', 'None')
                    lock_tgt = getattr(rm, 'friendly_lock_targets', {}).get(aid)
                    threat_sources = rm.get_rwr_threat_sources(aid)
                    
                    # RWR功能解释：0=无威胁，1=搜索，2=跟踪，3=锁定，4=导弹发射，5=导弹制导
                    threat_desc = {0: "无威胁", 1: "雷达搜索", 2: "雷达跟踪", 3: "雷达锁定", 4: "导弹发射", 5: "导弹制导"}
                    
                    logging.info(f"🛰️ [{aid}] RWR等级={lvl}({threat_desc.get(lvl, '未知')}) ECM={'ON' if ecm_on else 'OFF'}")
                    if ecm_on:
                        logging.info(f"    └─ ECM类型={ecm_type} | 威胁源={len(threat_sources)}个")
                    if lock_tgt:
                        logging.info(f"    └─ 雷达锁定={lock_tgt}")
                    
                    # ECM功能解释：干扰敌方雷达探测/锁定，降低被发现概率
                    if ecm_on and env.current_step % 900 == 0:
                        logging.info(f"💡 ECM功能: {aid}正在干扰敌方雷达，降低被探测/锁定概率")
            except Exception:
                pass
        
        # 记录数据（每10步记录一次）
        if env.current_step % 10 == 0:
            if hasattr(self.data_logger, 'record_step'):
                self.data_logger.record_step(env, current_time)
        
        # 每120秒打印一次状态信息
        if env.current_step % 600 == 0:
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    pos = env.agents[agent_id].get_position()
                    vel = env.agents[agent_id].get_velocity()
                    speed = np.linalg.norm(vel)
                    logging.info(f"[{agent_id}] 位置: ({pos[0]/1000:.1f}, {pos[1]/1000:.1f}, {pos[2]:.1f}m) 速度: {speed:.1f}m/s 高度: {pos[2]:.1f}m")
        
        # 获取观测、奖励、终止条件
        obs_dict = {}
        reward_dict = {}
        done_dict = {}
        info_dict = {}
        
        for agent_id in env.agents.keys():
            if not env.agents[agent_id].is_alive:
                obs_dict[agent_id] = np.zeros(self.observation_space.shape)
                reward_dict[agent_id] = 0.0
                done_dict[agent_id] = True
                info_dict[agent_id] = {}
                continue
            
            # 获取观测
            obs_dict[agent_id] = self.get_obs(env, agent_id)
            info_dict[agent_id] = {}
            
            # 检查终止条件
            term_result = self.get_termination(env, agent_id, info_dict[agent_id])
            if len(term_result) == 3:
                done, success, info = term_result
            else:
                done, info = term_result
                success = False
            
            done_dict[agent_id] = done
            reward_dict[agent_id] = 1.0 if env.agents[agent_id].is_alive else -100.0
        
        # 处理导弹发射（在所有agent处理完后）
        self._handle_missile_launches(env, current_time)
        
        # 检查是否结束，如果结束则保存数据
        if any(done_dict.values()):
            if hasattr(self.data_logger, 'save_all'):
                self.data_logger.save_all()
        
        # 环境期望字典格式
        return obs_dict, obs_dict, reward_dict, done_dict, info_dict
    
    # 保留原有的一些重要属性访问器（为了兼容性）
    @property
    def current_phase(self):
        """全局当前阶段"""
        return self.state_manager.current_phase
    
    @current_phase.setter
    def current_phase(self, phase):
        """设置全局当前阶段"""
        self.state_manager.current_phase = phase
    
    @property
    def agent_phases(self):
        """飞机阶段字典"""
        return self.state_manager.agent_phases
    
    @property
    def missile_launched(self):
        """导弹发射状态"""
        return self.state_manager.missile_launched
    
    @property
    def last_missile_launch_time(self):
        """最后导弹发射时间"""
        return self.state_manager.last_missile_launch_time
    
    @property
    def short_skate_states(self):
        """Short Skate状态"""
        return self.state_manager.short_skate_states
    
    @property
    def short_skate_start_time(self):
        """Short Skate开始时间"""
        return self.state_manager.short_skate_start_time
    
    @property
    def maneuver_states(self):
        """机动状态"""
        return self.state_manager.maneuver_states
    
    @property
    def lateral_split_targets(self):
        """横向分离目标"""
        return self.state_manager.lateral_split_targets
    
    @property
    def vertical_split_targets(self):
        """垂直分离目标"""
        return self.state_manager.vertical_split_targets
    
    def normalize_action(self, env, agent_id, action):
        """
        战术模板核心方法 - 动作归一化
        这是环境调用的主入口，将战术指令转换为底层控制
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        current_time = env.current_step * env.time_interval
        
        # 初始化agent状态
        if agent_id not in self.initial_heading:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
            self.initial_heading[agent_id] = np.rad2deg(current_heading)
            current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
            self.initial_altitude[agent_id] = current_altitude
            logging.info(f"{agent_id} 初始状态: 航向{self.initial_heading[agent_id]:.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")
        
        # 获取战术指令索引（调用get_action）
        action_result = self.get_action(env, agent_id)
        if action_result is None:
            logging.warning(f"⚠️ {agent_id} get_action返回None，使用默认动作")
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = 7, 8, 3
        elif not isinstance(action_result, tuple) or len(action_result) != 3:
            logging.warning(f"⚠️ {agent_id} get_action返回格式错误: {action_result}，使用默认动作")
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = 7, 8, 3
        else:
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = action_result
        
        # 使用baseline模型生成底层控制
        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
    
    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int) -> np.ndarray:
        """
        使用baseline模型生成底层控制
        完全按照原版实现
        """
        if self.baseline_model is None:
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        try:
            # 1. 获取原始观测
            raw_obs = self.get_obs(env, agent_id)
            
            # 2. 构造12维输入
            input_obs = np.zeros(12)
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)
            
            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            input_obs = np.expand_dims(input_obs, axis=0)
            
            # 3. 初始化RNN状态
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))
            
            # 4. 模型推理
            _action, _rnn_states = self.baseline_model(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            
            # 5. 归一化输出
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4
            
            # 6. 安全检查：低高度保护
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 3000:
                if current_alt < 1500:
                    norm_act[1] = max(norm_act[1], 0.5)
                    norm_act[3] = 1.0
                    if env.current_step % 60 == 0:
                        logging.warning(f"🛡️ [{agent_id}] 紧急拉升: 高度{current_alt:.0f}m < 1500m")
                else:
                    norm_act[1] = max(norm_act[1], 0.0)
                    norm_act[3] = max(norm_act[3], 0.8)
                    if env.current_step % 120 == 0:
                        logging.info(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < 3000m")
            
            return norm_act
            
        except Exception as e:
            logging.error(f"[{agent_id}] Baseline模型执行错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def _handle_missile_launches(self, env, current_time: float):
        """处理导弹发射逻辑 - 按原版逻辑"""
        # 调试：检查是否有发射请求（每步最多打印一次）
        launch_requests = [aid for aid, launched in self.state_manager.missile_launched.items() if launched]
        if launch_requests and self._last_launch_request_log_step != env.current_step:
            # 降低频率：每步最多一次
            self._last_launch_request_log_step = env.current_step
            if env.current_step % 300 == 0:  # 每60秒打印一次
                logging.info(f"🚀 导弹发射请求: {launch_requests}")
            
        # 遍历所有标记为需要发射导弹的agent
        for agent_id in list(self.state_manager.missile_launched.keys()):
            if not self.state_manager.missile_launched[agent_id]:
                continue
            
            # 检查agent是否还活着
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                self.state_manager.missile_launched[agent_id] = False
                continue
            
            # 检查冷却时间
            last_launch = self.state_manager.last_missile_launch_time.get(agent_id, -999)
            if current_time - last_launch < self.state_manager.missile_cooldown:
                continue
            
            # 检查导弹数量
            if env.agents[agent_id].num_missiles <= 0:
                logging.warning(f"⚠️ {agent_id} 导弹已用尽")
                self.state_manager.missile_launched[agent_id] = False
                continue
            
            # 寻找目标
            from core.target_assignment import get_target_with_fallback
            target_id = get_target_with_fallback(agent_id, env)
            target = env.agents.get(target_id) if target_id in env.agents else None
            
            if target is None or not target.is_alive:
                logging.warning(f"⚠️ {agent_id} 未找到有效目标 (目标ID:{target_id})")
                continue
            
            # 检查航向是否朝向敌机（防止背对发射）
            my_pos = env.agents[agent_id].get_position()
            target_pos = target.get_position()
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 计算目标方位角
            delta_x = target_pos[0] - my_pos[0]
            delta_y = target_pos[1] - my_pos[1]
            target_bearing = (np.rad2deg(np.arctan2(delta_y, delta_x)) + 360) % 360
            
            # 计算航向与目标方位的夹角
            heading_error = abs(self._normalize_angle_diff(target_bearing - current_heading))
            
            # 只有朝向敌机±120度范围内才允许发射（放宽以提高发射概率）
            if heading_error > 120:
                if env.current_step % 200 == 0:
                    logging.debug(f"⚠️ [{agent_id}] 发射条件不满足: 航向{current_heading:.0f}°, 目标方位{target_bearing:.0f}°, 夹角{heading_error:.0f}° >120°")
                continue
            
            # 在发射前执行完整条件检查
            current_phase = self.state_manager.get_agent_phase(agent_id)
            if not self.missile_manager.should_launch_missile(env, agent_id, current_phase, target_id):
                continue

            # 发射导弹
            self._launch_missile(env, agent_id, target, current_time)
            
            # 标记为已发射（防止重复发射）
            self.state_manager.missile_launched[agent_id] = False

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹 - 支持4枚导弹分批发射"""
        try:
            from envs.JSBSim.core.simulatior import MissileSimulator
            
            aircraft = env.agents[agent_id]
            
            # 创建导弹ID - 支持4枚导弹编号
            missile_count = 4 - aircraft.num_missiles + 1  # 1-4枚导弹
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count:0>2}"  # A100 → A10001, A10002, A10003, A10004
            
            # 使用引擎内置导弹工厂（由载荷/配置选择AIM-120C）
            missile = MissileSimulator.create(
                parent=aircraft,
                target=target,
                uid=missile_uid
            )
            
            # 添加到环境
            env.add_temp_simulator(missile)
            # 记录到环境导弹表，供威胁评估/RWR使用
            if not hasattr(env, 'missiles') or env.missiles is None:
                env.missiles = {}
            env.missiles[missile_uid] = missile
            
            # 记录导弹
            if not hasattr(self, 'missiles'):
                self.missiles = {}
            self.missiles[missile_uid] = missile
            
            # 更新发射时间和计数
            self.state_manager.last_missile_launch_time[agent_id] = current_time
            self.state_manager.missiles_fired[agent_id] += 1
            # 更新载机库存
            try:
                if hasattr(aircraft, 'num_left_missiles'):
                    aircraft.num_left_missiles = max(0, aircraft.num_left_missiles - 1)
                elif hasattr(aircraft, 'num_missiles'):
                    aircraft.num_missiles = max(0, aircraft.num_missiles - 1)
            except Exception:
                pass
            
            # 计算距离用于日志
            my_pos = aircraft.get_position()
            target_pos = target.get_position()
            distance = np.linalg.norm(np.array(target_pos) - np.array(my_pos))
            
            logging.info(f"🚀 [{agent_id}] 发射导弹 [{missile_uid}] → [{target.uid}] 距离{distance/1000:.1f}km")
            
        except Exception as e:
            logging.error(f"⚠️ [{agent_id}] 导弹发射失败: {e}")

    def _normalize_angle_diff(self, angle_diff):
        """标准化角度差到[-180, 180]区间"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    # ==================== 新增：二次进攻智能航向调整系统 ====================
    
    def _start_formation_reset_procedure(self, env, agent_id: str, current_time: float):
        """启动队形重置程序 - 新的二次进攻准备"""
        logging.info(f"🔄 {agent_id} 启动队形重置程序，准备二次进攻")
        
        # 通知战术执行器开始队形重置
        our_agents = [aid for aid in env.agents.keys() if aid.startswith('A') and env._jsbsims[aid].is_alive]
        for aid in our_agents:
            # 标记该飞机进入队形重置状态
            if not hasattr(self, 'formation_reset_agents'):
                self.formation_reset_agents = set()
            self.formation_reset_agents.add(aid)
            
        logging.info(f"📋 队形重置参与飞机: {list(our_agents)}")
    
    def _execute_formation_reset_procedure(self, env, agent_id: str) -> tuple:
        """执行队形重置程序"""
        # 首先执行队形重置机动
        maneuver_command = self.executor.execute_formation_reset(env, agent_id)

        # 然后检查是否准备好进行二次进攻
        if self.executor.check_second_attack_readiness(env):
            # 队形重置完成，开始统一的二次进攻
            self.selected_tactic = 'UNIFIED_SECOND_ATTACK'
            logging.info(f"✅ {agent_id} 队形重置完成，切换到统一二次进攻模式")
            return self.executor.execute_unified_second_attack(env, agent_id)
        else:
            # 继续执行队形重置机动
            return maneuver_command
    
    def _execute_unified_second_attack(self, env, agent_id: str) -> tuple:
        """执行统一的二次进攻"""
        return self.executor.execute_unified_second_attack(env, agent_id)
        
    def _execute_intelligent_reengagement(self, env, agent_id: str, current_time: float) -> tuple:
        """执行智能重新交战回转 - 动态航向调整"""
        if not hasattr(self, 'intelligent_reengagement_states'):
            return (7, 8, 3)  # 默认直飞
            
        state = self.intelligent_reengagement_states.get(agent_id)
        if not state:
            return (7, 8, 3)
            
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        target_heading = state['target_heading']
        
        # 计算航向差
        heading_diff = self._normalize_angle_diff(target_heading - current_heading)
        
        # 高度保护优先
        if current_altitude < 4000:
            alt_cmd = 8  # 轻微爬升
        elif current_altitude < 3000:
            alt_cmd = 9  # 爬升
        else:
            alt_cmd = 7  # 保持
            
        # 动态航向调整
        if abs(heading_diff) <= state['completion_threshold']:
            # 完成调整
            if not state['second_attack_ready']:
                state['second_attack_ready'] = True
                logging.info(f"✅ {agent_id} 智能回转完成！最终航向: {current_heading:.1f}° (目标0°)")
                
                # 选择二次进攻战术
                second_attack_tactic = self._select_second_attack_tactic()
                self.selected_tactic = second_attack_tactic
                self.is_second_attack = True
                logging.info(f"🎯 {agent_id} 开始二次进攻战术: {second_attack_tactic}")
                
                # 清理状态
                self.intelligent_reengagement_states.pop(agent_id, None)
            return (alt_cmd, 8, 3)  # 直飞
            
        # 进行航向调整
        if int(current_time) % 5 == 0:  # 每5秒输出一次调试信息
            logging.info(f"🔄 [{agent_id}] 动态调整: 当前{current_heading:.1f}° → 目标{target_heading}° (差值{heading_diff:.1f}°)")
            
        if abs(heading_diff) <= 10:
            # 接近目标，精调阶段
            logging.info(f"🎯 {agent_id} 航向接近目标({heading_diff:.1f}°)，进入精调阶段")
            if heading_diff > 0:
                return (alt_cmd, 10, 3)  # 轻微右转
            else:
                return (alt_cmd, 6, 3)   # 轻微左转
        elif abs(heading_diff) <= 45:
            # 中等调整
            if heading_diff > 0:
                return (alt_cmd, 12, 3)  # 中等右转
            else:
                return (alt_cmd, 4, 3)   # 中等左转
        else:
            # 大幅调整
            if heading_diff > 0:
                return (alt_cmd, 14, 3)  # 大幅右转
            else:
                return (alt_cmd, 2, 3)   # 大幅左转
                
    def _check_intelligent_reengagement_active(self, agent_id: str) -> bool:
        """检查是否正在执行智能重新交战"""
        if not hasattr(self, 'intelligent_reengagement_states'):
            return False
        return agent_id in self.intelligent_reengagement_states
        
    def _select_second_attack_tactic(self) -> str:
        """选择二次进攻战术 - 基于当前态势"""
        # 优先选择有效的编队战术
        available_tactics = ["DRAG_SHOOT", "PINCER_ATTACK", "HIGH_LOW_ATTACK"]
        
        # 如果是单机剩余，选择DRAG_SHOOT (单机也能执行)
        our_alive_count = sum(1 for aid in ['A0100', 'A0200'] if hasattr(self, 'env') and 
                             aid in self.env.agents and self.env.agents[aid].is_alive)
        
        if our_alive_count == 1:
            logging.info("🎯 单机剩余，选择DRAG_SHOOT战术")
            return "DRAG_SHOOT"
        
        # 双机编队时根据距离和态势选择
        # 简化策略：DRAG_SHOOT为默认，PINCER_ATTACK为备选
        import random
        selected = random.choice(["DRAG_SHOOT", "PINCER_ATTACK"])
        logging.info(f"🎯 二次进攻战术选择: {selected} (双机编队)")
        return selected
