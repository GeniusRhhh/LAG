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
from core.target_assignment import get_fixed_target, get_target_with_fallback
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


class TacticalPhase(Enum):
    """战术阶段枚举 - 范围阶段"""
    NLT_MELD = "NLT_MELD"        # 120-100km: 搜索目标&编队
    MELD_MTR = "MELD_MTR"        # 100-80km: 雷达融合&调整编队
    MTR_LR = "MTR_LR"            # 80-78km: 进入发射区
    LR_TR = "LR_TR"              # 78-75km: 发射&中制导
    TR_DOR = "TR_DOR"            # 75-70km: 中制导结束&规避
    DOR_DR = "DOR_DR"            # 70-65km: 规避机动
    DR_MAR = "DR_MAR"            # 65-40km: 脱离/重新进攻
    BEYOND_MAR = "BEYOND_MAR"    # <40km: WVR或完全脱离


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
        
        if env.current_step >= self.max_steps:
            self.log(f"达到时间限制: {env.current_step} 步")
            return True, False, info
        
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} 检测到极端状态")
            return True, False, info
        
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} 过载")
            return True, False, info
        
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]
        
        if len(red_alive) == 0:
            self.log("红方全灭")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            self.log("蓝方全灭")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}
        
        return False, False, info


class TacticalTask(MultipleCombatTask):
    """
    通用战术任务类（重构版）
    通过模块化架构将原4706行代码精简为~600行
    """
    
    def __init__(self, config, decision_manager=None):
        super().__init__(config)
        
        self.termination_conditions = [TacticalTermination(self.config)]
        
        # ===== 初始化模块化组件 =====
        # 1. 状态管理器
        self.state_manager = TacticalStateManager()
        
        # 2. 导弹管理器
        self.missile_manager = MissileManager(self.state_manager)
        
        # 3. 集成战术系统
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
        
        # ===== 保留的原有组件 =====
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False, "phase": None, "start_time": 0,
            "completed": False, "completed_time": 0
        }
        
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
        self.tactic_roles = {}
        self.decision_made = {'NLT': False, 'MELD': False, 'MTR': False}
        
        # 敌方AI
        self.enemy_ai = EnemyAIAdapter(config, project_name="tactical_simulation")
        logging.info("✅ 敌方AI系统初始化完成")
        
        # 导弹系统
        self.missile_system_enabled = True
        logging.info("✅ 导弹系统初始化完成")
        
        logging.info("TacticalTask初始化完成")
    
    def _load_baseline_model(self):
        """加载baseline模型"""
        model_path = get_root_dir() + '/model/baseline_model.pt'
        if os.path.exists(model_path):
            self.baseline_model = BaselineActor(
                obs_space=self.observation_space,
                act_space=self.action_space,
                model_path=model_path
            )
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
        self.selected_tactic = None
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
        
        # 重置雷达系统
        if self.radar_manager:
            self.radar_manager.reset()
        
        # 重置数据记录器
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
        
        # 检查导弹发射条件
        current_phase = self.state_manager.get_agent_phase(agent_id)
        if self.missile_manager.should_launch_missile(env, agent_id, current_phase):
            target_id = get_target_with_fallback(agent_id, env)
            self.missile_manager.execute_missile_launch(env, agent_id, target_id)
        
        # 敌方AI决策
        if agent_id.startswith('B'):
            return self._get_enemy_action(env, agent_id)
        
        # 我方决策：根据战术执行
        if self.selected_tactic is None:
            self._select_tactic_at_phase(env, agent_id)
        
        # 委托给战术执行器
        if self.selected_tactic == 'DRAG_SHOOT':
            return self.executor.execute_drag_shoot(env, agent_id)
        elif self.selected_tactic == 'PINCER_ATTACK':
            return self.executor.execute_pincer_attack(env, agent_id)
        elif self.selected_tactic == 'HIGH_LOW_ATTACK':
            return self.executor.execute_high_low_attack(env, agent_id)
        elif self.selected_tactic == 'FRONT_BACK':
            return self.executor.execute_front_back(env, agent_id)
        elif self.selected_tactic == 'SIDE_BY_SIDE':
            return self.executor.execute_side_by_side(env, agent_id)
        else:
            # 默认：平稳飞行
            return 7, 8, 3
    
    def _update_phase(self, env, agent_id: str, current_time: float):
        """更新飞机所处的战术阶段"""
        target_id = get_target_with_fallback(agent_id, env)
        if target_id is None:
            return
        
        target_aircraft = env._jsbsims.get(target_id)
        if target_aircraft is None or not target_aircraft.is_alive:
            return
        
        distance = TacticalUtils.calculate_distance_between(env.agents[agent_id], target_aircraft)
        
        # 根据距离判断阶段
        if distance >= self.tactical_distances['NLT']:
            new_phase = TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD']:
            new_phase = TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR']:
            new_phase = TacticalPhase.MTR_LR
        elif distance >= self.tactical_distances['LR']:
            new_phase = TacticalPhase.LR_TR
        elif distance >= self.tactical_distances['TR']:
            new_phase = TacticalPhase.TR_DOR
        elif distance >= self.tactical_distances['DOR']:
            new_phase = TacticalPhase.DOR_DR
        elif distance >= self.tactical_distances['DR']:
            new_phase = TacticalPhase.DR_MAR
        else:
            new_phase = TacticalPhase.BEYOND_MAR
        
        self.state_manager.set_agent_phase(agent_id, new_phase, current_time)
    
    def _select_tactic_at_phase(self, env, agent_id: str):
        """在关键节点选择战术"""
        current_phase = self.state_manager.get_agent_phase(agent_id)
        
        # 只在NLT节点做一次战术选择
        if current_phase == TacticalPhase.NLT_MELD and not self.decision_made['NLT']:
            logging.info("=" * 80)
            logging.info("🎯 [NLT节点] 开始战术决策（完整智能系统）")
            
            result = self.complete_tactical_system.select_tactic(
                env=env,
                my_agents=['A0100', 'A0200'],
                enemy_agents=['B0100', 'B0200']
            )
            
            self.selected_tactic = result['tactic']
            self.tactic_roles = result
            self.decision_made['NLT'] = True
            
            logging.info(f"   ✅ 选定战术: {self.selected_tactic}")
            logging.info(f"   ✅ 角色分配: 长机={result.get('lead')}, 僚机={result.get('wingman')}")
            logging.info("=" * 80)
    
    def _get_enemy_action(self, env, agent_id: str):
        """获取敌方行动（使用敌方AI）"""
        return self.enemy_ai.get_action(env, agent_id)
    
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
        """根据距离判断僚机阶段"""
        if distance >= self.tactical_distances['NLT']:
            return TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD']:
            return TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR']:
            return TacticalPhase.MTR_LR
        elif distance >= self.tactical_distances['LR']:
            return TacticalPhase.LR_TR
        elif distance >= self.tactical_distances['TR']:
            return TacticalPhase.TR_DOR
        elif distance >= self.tactical_distances['DOR']:
            return TacticalPhase.DOR_DR
        elif distance >= self.tactical_distances['DR']:
            return TacticalPhase.DR_MAR
        else:
            return TacticalPhase.BEYOND_MAR
    
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
