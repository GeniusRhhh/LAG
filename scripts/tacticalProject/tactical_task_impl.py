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
from tactical_termination import TacticalTermination

# 低层控制模型配置（可通过环境变量覆盖）
# FRIEND_BASELINE_MODEL: 我方使用的低层模型类型 ('F16' 或 'SU27')
# ENEMY_BASELINE_MODEL: 敌方使用的低层模型类型 ('F16' 或 'SU27')
FRIEND_BASELINE_MODEL = os.getenv("FRIEND_BASELINE_MODEL", "F16").upper()
ENEMY_BASELINE_MODEL = os.getenv("ENEMY_BASELINE_MODEL", "F16").upper()

# 导入核心模块
from core.threat_evaluator_complete import CompleteThreatEvaluator
from core.situation_evaluator import SituationEvaluator
from core.intent_predictor import IntentPredictor
from core.decision_table import DecisionTable
from core.tactic_selector_v2 import TacticSelectorV2
from core.complete_tactical_system import CompleteTacticalSystem
from core.target_assignment import get_fixed_target, get_enemy_team
from tactical_types import TacticalPhase, get_target_with_fallback
from enemy_ai_adapter import EnemyAIAdapter
from utils.data_logger import DataLogger
from utils.constants import CONTROL_RANGES
from cap.run_logging import get_sim_log_prefix

# 导入新重构的模块
from tactical_state_manager import TacticalStateManager
from missile_manager import MissileManager
from tactical_decision_maker import TacticalDecisionMaker
from tactical_executor import TacticalExecutor
from maneuver_library import ManeuverLibrary
from tactical_utils import TacticalUtils
from core.node_decision_maker import NodeDecisionMaker, NodeContext
from simulation.radar_manager import update_all_radars, get_friendly_radar_state, get_enemy_radar_state, get_unified_radar_manager
from core.tactical_selector_algorithm import TacticalSelectorAlgorithm
import tactical_task_refactor_helpers as _ttrfh
import tactical_task_reengage_helpers as _ttrh
from tactical_task_action_mixin import TacticalTaskActionMixin
from core.situation_algorithm_switcher import get_situation_algorithm_switcher

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


class TacticalTask(TacticalTaskActionMixin, MultipleCombatTask):
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
        # 1. 状态管理器 - 每个战术任务有自己独立的实例
        self.state_manager = TacticalStateManager()
        
        # 2. 导弹管理器
        self.missile_manager = MissileManager(self.state_manager)
        
        # 3. 返航状态跟踪
        self.returning_agents = set()  # 正在返航的智能体
        
        # 4. 阶段锁定机制 - 防止阶段回退
        self.phase_lock = {}  # agent_id -> locked_phase (一旦进入某些关键阶段就不允许回退)
        
        my_intent_str = getattr(config, 'friendly_intent', 'CONSERVATIVE_CLEAR')
        # Legacy IntegratedTacticalSystem used the old rule-based intent path.
        # Keep the attribute for compatibility, but route runtime decisions only
        # through CompleteTacticalSystem + SituationAlgorithmSwitcher.
        self.integrated_system = None
        self.complete_tactical_system = CompleteTacticalSystem(my_intent=my_intent_str)
        logging.info(f"[集成战术系统] 已启动 (我方意图: {my_intent_str})")
        self.my_intent_str = my_intent_str
        
        # 4. 决策制定器（需要威胁和态势评估器）
        self.threat_evaluator = CompleteThreatEvaluator()
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
        
        # 🔥 修复3: 使用高级编队重整管理器
        from advanced_formation_reset_manager import AdvancedFormationResetManager
        self.formation_reset_manager = AdvancedFormationResetManager()
        self._formation_reset_managers = {
            'A0100/A0200': AdvancedFormationResetManager(),
            'A0300/A0400': AdvancedFormationResetManager(),
        }
        self._formation_tactic_overrides = {}
        # 编队长僚机首次判定锁：首次确定后在双机都存活期间保持不变
        self._formation_role_lock = {}
        self._dr_reengage_decision_made = {}
        self._dr_retreat_decision_made = {}
        self._second_attack_tactic_selected_by_formation = {}
        
        # 绑定节点决策器依赖
        try:
            # 使用完整系统中的评估器与选择器，避免方法缺失（detect_incoming_missiles等）
            self.node_decision_maker.decision_table = self.complete_tactical_system.decision_table
            self.node_decision_maker.tactical_selector = self.complete_tactical_system.tactical_selector
            self.node_decision_maker.threat_evaluator = self.complete_tactical_system.threat_evaluator
            self.node_decision_maker.situation_evaluator = self.situation_evaluator
            
            # Runtime is locked to bvr_intent_new only.
            self.situation_algorithm_switcher = get_situation_algorithm_switcher()  # 修复：函数不接受参数
            # ✅ 启动时打印算法状态，便于确认是否真正使用算法1（以及权重是否就绪/是否会回退）
            try:
                status = self.situation_algorithm_switcher.get_algorithm_status()
                cur = status.get("current_algorithm")
                ready = status.get("algorithms", {}).get(cur, {}).get("ready", None)
                logging.info(f"[意图识别算法] current={cur}, ready={ready}")
            except Exception:
                pass
            # 兼容性：保留intent_recognizer属性，但统一指向算法切换器。
            self.node_decision_maker.intent_recognizer = self.situation_algorithm_switcher
            
            # 🔥 修复：注释掉不存在的方法调用
            # logging.info(f"🔧 态势识别算法: {self.situation_algorithm_switcher.get_current_algorithm()}")
        except Exception as e:
            logging.warning(f"节点决策器依赖绑定失败: {e}")
            self.situation_algorithm_switcher = None
        
        # ===== 保留的原有组件 =====
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False, "phase": None, "start_time": 0,
            "completed": False, "completed_time": 0
        }
        if hasattr(self, 'executor'):
            self.executor._heading_smoother = {}
        # 二次进攻状态：按机独立（兼容保留 self.is_second_attack 作为聚合标志）
        self.agent_second_attack = {}
        self.is_second_attack = False
        # 🔥 修复：初始化导弹计数器
        self.aircraft_missile_counts = {}
        self._pending_second_missile = {}  # 延迟发射的第二枚导弹标记
        # 决策链路追踪（用于论文/答辩解释）：避免重复刷屏
        self._decision_trace_last_step = {}  # (agent_id,node) -> step
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
        logging.info("[数据记录器] 初始化完成")
        
        # 雷达系统
        if RADAR_AVAILABLE:
            try:
                self.radar_manager = UnifiedRadarManager()
                logging.info("[雷达系统] 初始化完成 (APG-68 vs N001VE)")
            except Exception as e:
                logging.warning(f"雷达系统初始化失败: {e}")
                self.radar_manager = None
        else:
            self.radar_manager = None

        # 低层控制模型（F16 Baseline + SU27 PPO，可按阵营切换）
        self._load_baseline_model()
        self.friend_lowlevel_type = FRIEND_BASELINE_MODEL
        self.enemy_lowlevel_type = ENEMY_BASELINE_MODEL
        logging.info(f"[低层控制模型] 配置: 我方={self.friend_lowlevel_type}, 敌方={self.enemy_lowlevel_type}")

        # CAP区域约束
        self.cap_boundary = {
            'type': 'racetrack', 'center_x': 0, 'center_y': 0,
            'length': 240000, 'width': 120000, 'z_min': 3000, 'z_max': 13000,
            'rect_x_min': -60000, 'rect_x_max': 60000,
            'rect_y_min': -60000, 'rect_y_max': 60000,
            'circle_radius': 60000,
            'south_circle_center': (-60000, 0),
            'north_circle_center': (60000, 0),
        }
        logging.info("[CAP区域] 南北向跑马道（已扩大）")
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
        # 🔧 [任务3修复] 增加LR-TR缓冲区，防止78km附近的阶段震荡
        self.tactical_distances = getattr(config, 'tactical_distances', {
            'NLT': 120000, 'MELD': 100000, 'MTR': 80000,
            'LR': 82000, 'TR': 75000, 'DOR': 70000,  # LR从78000增加到82000，创建7km缓冲区
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
        logging.info("[敌方AI系统] 初始化完成")
        
        # 导弹系统
        self.missile_system_enabled = True

        # ===== StateManager字典通过@property访问器提供（见文件末尾）=====
        # 不需要在此处赋值，因为已有@property装饰器提供访问

        # 日志节流
        self._last_launch_request_log_time = 0.0
        self._last_launch_request_log_step = -1

        logging.info("[导弹系统] 初始化完成")
        logging.info("TacticalTask初始化完成")

    def _format_sim_timestamp(self, env=None, current_time: float = None) -> str:
        sim_time = current_time
        step = None
        if sim_time is None and env is not None:
            try:
                step = int(getattr(env, 'current_step', 0))
                sim_time = float(step) * float(getattr(env, 'time_interval', 0.2))
            except Exception:
                sim_time = None
        if step is None and env is not None:
            try:
                step = int(getattr(env, 'current_step', 0))
            except Exception:
                step = 0
        if sim_time is None:
            sim_time = 0.0
        return get_sim_log_prefix(step=step, time_s=float(sim_time))

    def _log_key_event(
        self,
        env,
        agent_id: str,
        title: str,
        *,
        current_time: float = None,
        level: int = logging.INFO,
        **fields,
    ) -> None:
        prefix = self._format_sim_timestamp(env, current_time=current_time)
        segments = [f"{k}={v}" for k, v in fields.items() if v is not None]
        message = f"{prefix} [{title}] [{agent_id}]"
        if segments:
            message = f"{message} " + " | ".join(segments)
        logging.log(level, message)
    
    def _load_baseline_model(self):
        """加载低层控制模型

        - F16: BaselineActor + baseline_model.pt
        - SU27: PPOActor  + su27_baseline.pt

        两种模型都会尝试加载，我方/敌方可以在 friend_lowlevel_type / enemy_lowlevel_type 中选择使用哪一个。
        """
        root_dir = get_root_dir()
        f16_path = os.path.join(root_dir, 'model', 'baseline_model.pt')
        # su27_path = os.path.join(root_dir, 'model', 'su27_baseline.pt')
        # 使用用户指定的特定模型路径
        su27_path = r"D:\Pycharm\LAG\lqyLAG\scripts\results\SingleControl\1\heading_su27\ppo\su27_baseline_v1\run40\actor_990.pt"

        # F16 BaselineActor（原始模型）
        self.baseline_model = None
        if os.path.exists(f16_path):
            try:
                self.baseline_model = BaselineActor(input_dim=12, use_mlp_actlayer=False)
                state_dict = torch.load(f16_path, map_location='cpu', weights_only=True)
                self.baseline_model.load_state_dict(state_dict)
                self.baseline_model.eval()
                logging.info(f"[低层模型] F16 Baseline 加载成功: {f16_path}")
            except Exception as e:
                logging.error(f"[低层模型] F16 Baseline 加载失败: {e}")
                self.baseline_model = None
        else:
            logging.warning(f"[低层模型] 未找到 F16 Baseline 模型文件: {f16_path}")

        # SU27 Baseline（PPOActor）
        self.su27_baseline_actor = None
        if os.path.exists(su27_path):
            try:
                self.su27_baseline_actor = BaselineActor(input_dim=12, use_mlp_actlayer=True)
                state_dict = torch.load(su27_path, map_location=torch.device('cpu'), weights_only=True)
                
                # 检测是否需要 use_mlp_actlayer
                use_mlp = any(k.startswith('act.mlp') for k in state_dict.keys())
                if not use_mlp:
                     self.su27_baseline_actor = BaselineActor(input_dim=12, use_mlp_actlayer=False)
                
                self.su27_baseline_actor.load_state_dict(state_dict)
                self.su27_baseline_actor.eval()
                logging.info(f"[低层模型] SU27 Baseline(PPO) 加载成功: {su27_path}")
            except Exception as e:
                logging.error(f"[低层模型] SU27 Baseline(PPO) 加载失败，将仅使用 F16 Baseline: {e}")
                self.su27_baseline_actor = None
        else:
            logging.warning(f"[低层模型] 未找到 SU27 Baseline 模型文件: {su27_path}")
    
    def reset(self, env):
        """重置任务状态"""
        self._agent_die_flag = {}
        self.agent_ids = list(env.agents.keys())
        self.step_count = 0
        self.allocation_counter = 0
        self._target_allocation = {}
        self._friendly_sim_recreate_state = {}
        self._enemy_sim_recreate_state = {}

        if self.use_baseline:
            try:
                self.baseline_agent.reset()
            except Exception:
                pass

        for func in getattr(self, 'reward_functions', []):
            if hasattr(func, 'reset'):
                try:
                    func.reset(self, env)
                except Exception:
                    pass

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
            logging.info(f"[强制战术] 重置时保持: {self.force_tactic}")
            self.selected_tactic = self.force_tactic
        self.tactic_roles = {}
        self._formation_role_lock = {}
        self.decision_made = {'NLT': False, 'MELD': False, 'MTR': False}
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False, "phase": None, "start_time": 0,
            "completed": False, "completed_time": 0
        }

        # 重置二次进攻状态（按机独立 + 聚合标志）
        self.agent_second_attack = {}
        self.is_second_attack = False
        self.env = env
        self._friendly_initial_north_m = {}
        self._friendly_initial_east_m = {}
        
        # 初始化每架飞机的状态
        for agent_id in env.agents.keys():
            if env.agents[agent_id].is_alive:
                self.state_manager.init_agent_states(agent_id, max_missiles=4)
                self.initial_heading[agent_id] = env.agents[agent_id].get_property_value(c.attitude_psi_rad)
                self.initial_altitude[agent_id] = env.agents[agent_id].get_property_value(c.position_h_sl_m)
                self._inner_rnn_states[agent_id] = torch.zeros(1, 1, 128)
                pos = env.agents[agent_id].get_position()
                if agent_id.startswith('A'):
                    self._friendly_initial_north_m[agent_id] = float(pos[0])
                    self._friendly_initial_east_m[agent_id] = float(pos[1])
                logging.info(f"{agent_id} 初始状态: 航向{np.rad2deg(self.initial_heading[agent_id]):.1f}°, 高度{self.initial_altitude[agent_id]:.1f}m")
                self.agent_second_attack[agent_id] = False
        
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

    def step(self, env):
        """2v2战术任务的单步汇总接口。"""
        try:
            try:
                self.step_count = int(getattr(env, 'current_step', getattr(self, 'step_count', 0) + 1))
            except Exception:
                self.step_count = int(getattr(self, 'step_count', 0)) + 1

            current_time = float(self.step_count) * float(getattr(env, 'time_interval', 0.2))

            try:
                update_all_radars(env, current_time)
            except Exception as e:
                logging.debug(f"[TacticalTask.step] radar update skipped: {e}")

            agent_ids = list(getattr(env, 'agents', {}).keys())
            obs = {agent_id: self.get_obs(env, agent_id) for agent_id in agent_ids}

            sorted_agent_ids = sorted(agent_ids)
            all_obs = np.stack([obs[agent_id] for agent_id in sorted_agent_ids], axis=0)
            shared_flat = all_obs.flatten()
            share_obs = {agent_id: shared_flat.copy() for agent_id in agent_ids}

            rewards = {}
            dones = {}
            infos = {}

            for agent_id in agent_ids:
                info = {}
                reward_sum = 0.0
                reward_details = {}

                try:
                    state_dict = self.get_state_dict(env, agent_id)
                except Exception:
                    state_dict = {}

                for func in getattr(self, 'reward_functions', []):
                    try:
                        func_name = func.__class__.__name__
                        if func_name in ("TacticalRewardNew", "RadarLockRewardNew", "MissileHitRewardNew"):
                            reward_info = func.get_reward(self, env, agent_id, state_dict)
                        else:
                            reward_info = func.get_reward(self, env, agent_id)

                        if isinstance(reward_info, (tuple, list)) and len(reward_info) > 0:
                            reward_value = reward_info[0]
                        else:
                            reward_value = reward_info if isinstance(reward_info, (int, float, np.floating)) else 0.0

                        reward_details[func_name] = float(reward_value)
                        reward_sum += float(reward_value)
                    except Exception as e:
                        reward_details[func.__class__.__name__] = 0.0
                        logging.debug(f"[TacticalTask.step] reward component failed for {agent_id}: {func.__class__.__name__}: {e}")

                done, info = self.get_termination(env, agent_id, info)

                try:
                    phase_obj = self.state_manager.get_agent_phase(agent_id)
                    phase_name = getattr(phase_obj, 'value', str(phase_obj))
                except Exception:
                    phase_name = "UNKNOWN"

                try:
                    current_tactic = self._get_agent_tactic(agent_id)
                except Exception:
                    current_tactic = None

                try:
                    current_target = self.state_manager.get_target(agent_id)
                except Exception:
                    current_target = None

                if not isinstance(info, dict):
                    info = {}
                info.update({
                    "step": self.step_count,
                    "time": current_time,
                    "current_phase": phase_name,
                    "current_tactic": current_tactic,
                    "current_target": current_target,
                    "is_alive": bool(getattr(env.agents.get(agent_id), 'is_alive', False)),
                    "reward_details": reward_details,
                })

                rewards[agent_id] = np.array([float(np.clip(reward_sum, -10.0, 10.0))])
                dones[agent_id] = [bool(done)]
                infos[agent_id] = info

            return obs, share_obs, rewards, dones, infos

        except Exception as e:
            logging.error(f"Critical error in TacticalTask.step: {e}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")

            agent_ids = list(getattr(env, 'agents', {}).keys())
            obs_length = int(getattr(self, 'obs_length', 32))
            obs = {agent_id: np.zeros(obs_length) for agent_id in agent_ids}
            share_obs = {agent_id: np.zeros(obs_length * max(1, len(agent_ids))) for agent_id in agent_ids}
            rewards = {agent_id: np.array([0.0]) for agent_id in agent_ids}
            dones = {agent_id: [False] for agent_id in agent_ids}
            infos = {agent_id: {} for agent_id in agent_ids}
            return obs, share_obs, rewards, dones, infos

    def is_agent_second_attack(self, agent_id: str) -> bool:
        return bool(getattr(self, 'agent_second_attack', {}).get(agent_id, False))

    def set_agent_second_attack(self, agent_id: str, value: bool = True):
        if not hasattr(self, 'agent_second_attack') or self.agent_second_attack is None:
            self.agent_second_attack = {}
        self.agent_second_attack[agent_id] = bool(value)
        # 兼容：保留全局标志，表示是否有任意友机处于二次进攻
        try:
            self.is_second_attack = any(bool(v) for v in self.agent_second_attack.values())
        except Exception:
            self.is_second_attack = bool(value)

    def _deescalate_red_formation_to_defense(
        self,
        env,
        agent_id: str,
        reason: str = "",
        clear_returning: bool = True,
    ):
        formation_agents = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        if not formation_agents and self._is_agent_alive_in_any_collection(env, agent_id):
            formation_agents = [agent_id]

        formation_label = self._get_formation_label(agent_id, env) if hasattr(self, '_get_formation_label') else agent_id
        executor = getattr(self, 'executor', None)
        state_manager = getattr(self, 'state_manager', None)
        current_time = float(getattr(env, "current_step", 0)) * float(getattr(env, "time_interval", 0.2))
        guard_release_until = current_time + 8.0

        for aid in formation_agents:
            if clear_returning and hasattr(self, 'returning_agents'):
                self.returning_agents.discard(aid)
            self.set_agent_second_attack(aid, False)
            if hasattr(self, 'missile_launched'):
                self.missile_launched[aid] = False
            if state_manager and hasattr(state_manager, 'missile_launched'):
                state_manager.missile_launched[aid] = False
            if hasattr(self, '_selected_second_attack_tactic'):
                self._selected_second_attack_tactic.pop(aid, None)
            if hasattr(self, 'formation_reset_agents'):
                self.formation_reset_agents.discard(aid)
            if executor is not None and hasattr(executor, 'tactical_turn_states'):
                executor.tactical_turn_states.pop(aid, None)
            if hasattr(self, '_adaptive_bvr_standoff_latch'):
                self._adaptive_bvr_standoff_latch[aid] = True
            if hasattr(self, '_red_guard_release_until'):
                self._red_guard_release_until[aid] = max(
                    float(self._red_guard_release_until.get(aid, -999.0)),
                    guard_release_until,
                )
            self._set_agent_tactic(aid, 'DEFENSIVE_GUARD')

        formation_manager = self._get_formation_reset_manager(agent_id) if hasattr(self, '_get_formation_reset_manager') else None
        if formation_manager is not None:
            try:
                formation_manager.active = False
                formation_manager.phase = 'inactive'
                formation_manager.active_agents = []
                formation_manager.zero_heading_hold_start = None
            except Exception:
                pass

        if hasattr(self, '_second_attack_tactic_selected_by_formation'):
            self._second_attack_tactic_selected_by_formation.pop(formation_label, None)
        return formation_agents

    def _preserve_red_defense_if_needed(
        self,
        env,
        agent_id: str,
        *,
        current_phase_name: str = "",
        reason: str = "",
        log_tag: str = "DEFENSE_INTERCEPT",
        clear_returning: bool = True,
    ):
        if not str(agent_id).startswith('A'):
            return None

        try:
            if not self._has_alive_enemy_anywhere(env, agent_id):
                return None
        except Exception:
            return None

        try:
            posture = self._evaluate_red_defensive_posture(
                env,
                agent_id,
                current_phase_name=current_phase_name,
                reason=reason or current_phase_name or log_tag,
            )
        except Exception:
            return None

        if not posture.get("preserve", False):
            return None

        preserve_reason = str(posture.get("reason", reason or current_phase_name or log_tag))
        self._deescalate_red_formation_to_defense(
            env,
            agent_id,
            reason=preserve_reason,
            clear_returning=clear_returning,
        )

        if getattr(env, "current_step", 0) % 80 == 0:
            formation_label = self._get_formation_label(agent_id, env)
            logging.info(
                "🛡️ [%s:%s] %s zone=%s dist=%.1fkm my_depth=%.1fkm target_depth=%.1fkm "
                "intent=%s/%s retreat=%d phase=%s grp=%s dyn=%d closure=%.1fm/s away=%.1fdeg guidance=%d reason=%s",
                formation_label,
                agent_id,
                log_tag,
                posture.get("target_zone", "UNKNOWN"),
                float(posture.get("distance_km", float('nan'))),
                float(posture.get("my_depth_km", float('nan'))),
                float(posture.get("target_depth_km", float('nan'))),
                posture.get("enemy_intent", "UNKNOWN"),
                posture.get("enemy_intent_source", "unknown"),
                int(posture.get("enemy_wave_retreating", False)),
                posture.get("enemy_group_phase", "UNKNOWN"),
                posture.get("enemy_group_id", ""),
                int(posture.get("enemy_group_dynamic", False)),
                float(posture.get("closure_m_s", float('nan'))),
                float(posture.get("retreat_heading_error_deg", float('nan'))),
                int(posture.get("guidance_commit", False)),
                preserve_reason,
            )
        return posture

    def _build_red_reengage_override_plan(
        self,
        env,
        agent_id: str,
        *,
        current_phase_name: str = "",
        reason: str = "",
        allow_template_reset: bool = True,
    ) -> dict:
        posture = self._evaluate_red_defensive_posture(
            env,
            agent_id,
            current_phase_name=current_phase_name,
            reason=reason or current_phase_name or "reengage_override",
        )
        formation_agents = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        current_time = float(getattr(env, "current_step", 0)) * float(getattr(env, "time_interval", 0.2))
        guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
        guard_release_active = current_time < guard_release_until
        formation_alive = len(formation_agents)
        missiles_left = int(posture.get("missiles_left", 0) or 0)
        formation_missiles = int(posture.get("formation_missiles", 0) or 0)
        release_for_engage = bool(posture.get("release_for_engage", False))
        enemy_wave_retreating = bool(posture.get("enemy_wave_retreating", False))
        target_zone = str(posture.get("target_zone", "UNKNOWN") or "UNKNOWN").upper()
        distance_km = float(posture.get("distance_km", float('nan')))
        forward_commit_state = {}
        try:
            forward_commit_state = self._evaluate_red_forward_commit_block(
                env,
                agent_id,
                distance=None if not np.isfinite(distance_km) else distance_km * 1000.0,
                current_phase_name=current_phase_name,
                guidance_commit=bool(posture.get("guidance_commit", False)),
            )
        except Exception:
            forward_commit_state = {}
        forward_commit_block = bool(forward_commit_state.get("block", False))
        replacement_tactic = 'ADAPTIVE_ATTACK'

        if guard_release_active:
            replacement_tactic = 'DEFENSIVE_GUARD'
        elif bool(posture.get("preserve", False)) or missiles_left <= 0 or formation_missiles <= 0:
            replacement_tactic = 'DEFENSIVE_GUARD'
        else:
            template_ready = False
            if allow_template_reset and formation_alive >= 2 and current_phase_name not in ('DOR_DR', 'DR_MAR', 'BEYOND_MAR'):
                try:
                    template_ready = bool(self._should_use_template_tactic(env, agent_id, 'UNIFIED_SECOND_ATTACK'))
                except Exception:
                    template_ready = False
            if forward_commit_block:
                replacement_tactic = 'DEFENSIVE_GUARD'
            elif formation_alive <= 1 and not release_for_engage:
                replacement_tactic = 'DEFENSIVE_GUARD'
            elif template_ready:
                replacement_tactic = 'FORMATION_RESET'
            elif release_for_engage:
                replacement_tactic = 'ADAPTIVE_ATTACK'
            elif enemy_wave_retreating and (not np.isfinite(distance_km) or distance_km >= 55.0):
                replacement_tactic = 'DEFENSIVE_GUARD'
            elif target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN') and (not np.isfinite(distance_km) or distance_km >= 50.0):
                replacement_tactic = 'DEFENSIVE_GUARD'

        return {
            "posture": posture,
            "formation_agents": formation_agents,
            "replacement_tactic": replacement_tactic,
            "preserve_defense": replacement_tactic == 'DEFENSIVE_GUARD',
            "forward_commit_block": forward_commit_block,
            "forward_commit_reason": str(forward_commit_state.get("reason", "")),
            "guard_release_active": guard_release_active,
        }

    def _sort_friendly_by_initial_lateral(self, agent_ids):
        return sorted(
            list(agent_ids),
            key=lambda aid: float(getattr(self, '_friendly_initial_east_m', {}).get(aid, 0.0)),
        )

    def _get_reference_env(self, env=None):
        if env is not None and hasattr(env, 'resolve_real_agent_id'):
            return env

        candidates = []

        def _add(candidate):
            if candidate is None:
                return
            if any(candidate is existing for existing in candidates):
                return
            candidates.append(candidate)

        cap_task_sink = getattr(self, '_cap_task_sink', None)
        _add(env)
        if env is not None:
            _add(getattr(env, '_original', None))
        _add(getattr(cap_task_sink, 'env', None) if cap_task_sink is not None else None)
        _add(getattr(self, 'env', None))

        best_env = None
        best_score = -1
        for candidate in candidates:
            try:
                score = max(
                    len(getattr(candidate, 'agents', {}) or {}),
                    len(getattr(candidate, '_jsbsims', {}) or {}),
                )
            except Exception:
                score = 0
            if score > best_score:
                best_env = candidate
                best_score = score
        return best_env

    def _is_bridge_isolated_env(self, env) -> bool:
        return bool(getattr(env, "_bridge_isolated", False))

    def _resolve_real_agent_id(self, env, agent_id: str) -> str:
        resolved_id = str(agent_id or "")
        if env is not None and hasattr(env, "resolve_real_agent_id"):
            try:
                candidate = env.resolve_real_agent_id(resolved_id)
                if candidate:
                    resolved_id = str(candidate)
            except Exception:
                pass
        return resolved_id

    def _resolve_friendly_lookup_ids(self, env, agent_id: str) -> list[str]:
        lookup_ids = []

        def _append(candidate):
            cid = str(candidate or "").strip()
            if cid and cid not in lookup_ids:
                lookup_ids.append(cid)

        _append(self._resolve_real_agent_id(env, agent_id))
        _append(agent_id)
        return lookup_ids

    def _get_cap_enemy_group_snapshot(self, env, enemy_agent_id: str) -> dict:
        cap_task_sink = getattr(self, "_cap_task_sink", None)
        enemy_adapter = getattr(cap_task_sink, "enemy_adapter", None) if cap_task_sink is not None else None
        if enemy_adapter is None or not hasattr(enemy_adapter, "get_group_snapshot"):
            return {}
        try:
            return enemy_adapter.get_group_snapshot(self._resolve_real_agent_id(env, enemy_agent_id)) or {}
        except Exception:
            return {}

    def _evaluate_cap_enemy_north_phase_status(
        self,
        env,
        enemy_agent_id: str,
        target=None,
        distance: float = None,
        target_zone: str = "",
        target_depth_km: float = float('nan'),
    ) -> dict:
        cap_task_sink = getattr(self, "_cap_task_sink", None)
        enemy_adapter = getattr(cap_task_sink, "enemy_adapter", None) if cap_task_sink is not None else None
        if enemy_adapter is None or not hasattr(enemy_adapter, "evaluate_north_phase_status"):
            return {}

        zone_name = str(target_zone or "UNKNOWN")
        depth_km = float(target_depth_km) if np.isfinite(target_depth_km) else float('nan')
        if target is not None:
            if zone_name == "UNKNOWN":
                try:
                    zone_name = self._classify_target_risk_zone(target, env=env)
                except Exception:
                    zone_name = "UNKNOWN"
            if not np.isfinite(depth_km):
                try:
                    depth_km = float(self._get_aircraft_battlefield_depth_km(env, target))
                except Exception:
                    depth_km = float('nan')

        distance_km = float(distance) / 1000.0 if distance is not None and np.isfinite(distance) else float('nan')
        try:
            return enemy_adapter.evaluate_north_phase_status(
                self._resolve_real_agent_id(env, enemy_agent_id),
                target_y_km=depth_km,
                target_zone=zone_name,
                distance_km=distance_km,
            ) or {}
        except Exception:
            return {}

    def _resolve_friendly_formations(self, env):
        env = self._get_reference_env(env)
        if env is None:
            return []
        if self._is_bridge_isolated_env(env):
            visible_friendlies = self._dedupe_friendly_members(
                self._get_bridge_visible_alive_agent_ids(env, 'A')
            )
            if len(visible_friendlies) >= 2:
                return [tuple(visible_friendlies[:2])]
            return [(aid,) for aid in visible_friendlies]
        alive_ids = set()
        for aid in ('A0100', 'A0200', 'A0300', 'A0400'):
            if self._is_agent_alive_in_any_collection(env, aid):
                alive_ids.add(aid)
        left_alive = [aid for aid in ('A0100', 'A0200') if aid in alive_ids]
        right_alive = [aid for aid in ('A0300', 'A0400') if aid in alive_ids]

        groups = []
        full_pairs = []
        leftovers = []

        if len(left_alive) == 2:
            full_pairs.append(tuple(self._sort_friendly_by_initial_lateral(left_alive)))
        else:
            leftovers.extend(left_alive)

        if len(right_alive) == 2:
            full_pairs.append(tuple(self._sort_friendly_by_initial_lateral(right_alive)))
        else:
            leftovers.extend(right_alive)

        groups.extend(full_pairs)
        leftovers = self._sort_friendly_by_initial_lateral(leftovers)

        if (not full_pairs) and len(leftovers) >= 2:
            groups.append(tuple(leftovers[:2]))
            leftovers = leftovers[2:]

        for aid in leftovers:
            groups.append((aid,))
        return groups

    def _make_friendly_group_label(self, members):
        member_set = set(members)
        if member_set == {'A0100', 'A0200'}:
            return 'A0100/A0200'
        if member_set == {'A0300', 'A0400'}:
            return 'A0300/A0400'
        if len(members) == 2:
            return f"A_REFORMED[{'+'.join(members)}]"
        return f"SOLO_{members[0]}"

    def _iter_agent_collections(self, env):
        seen = set()

        def _yield_collection(collection):
            if not isinstance(collection, dict):
                return
            collection_id = id(collection)
            if collection_id in seen:
                return
            seen.add(collection_id)
            yield collection

        reference_env = self._get_reference_env(env)
        if self._is_bridge_isolated_env(reference_env) or self._is_bridge_isolated_env(env):
            isolated_env = reference_env if self._is_bridge_isolated_env(reference_env) else env
            if isolated_env is None:
                return
            for collection in _yield_collection(getattr(isolated_env, 'agents', None)):
                yield collection
            for collection in _yield_collection(getattr(isolated_env, '_jsbsims', None)):
                yield collection
            return
        if reference_env is None and env is None:
            return
        original_env = getattr(env, '_original', None) if env is not None else None
        env_chain = []
        for env_obj in (reference_env, env, original_env):
            if env_obj is None:
                continue
            if any(env_obj is existing for existing in env_chain):
                continue
            env_chain.append(env_obj)

        for env_obj in env_chain:
            for collection in _yield_collection(getattr(env_obj, 'agents', None)):
                yield collection
            for collection in _yield_collection(getattr(env_obj, '_jsbsims', None)):
                yield collection

    def _iter_status_env_views(self, env):
        seen = set()
        cap_task_sink = getattr(self, '_cap_task_sink', None)
        reference_env = self._get_reference_env(env)
        if self._is_bridge_isolated_env(reference_env) or self._is_bridge_isolated_env(env):
            isolated_env = reference_env if self._is_bridge_isolated_env(reference_env) else env
            if isolated_env is not None:
                yield "bridge_isolated", isolated_env
            return
        original_env = getattr(env, '_original', None) if env is not None else None
        sink_env = getattr(cap_task_sink, 'env', None) if cap_task_sink is not None else None
        task_env = getattr(self, 'env', None)
        reference_original = getattr(reference_env, '_original', None) if reference_env is not None else None

        def _add(name, env_obj):
            if env_obj is None:
                return
            env_id = id(env_obj)
            if env_id in seen:
                return
            seen.add(env_id)
            yield name, env_obj

        for item in _add("env_original", original_env):
            yield item
        for item in _add("cap_sink", sink_env):
            yield item
        for item in _add("task_env", task_env):
            yield item
        for item in _add("env", env):
            yield item
        for item in _add("reference", reference_env):
            yield item
        for item in _add("reference_original", reference_original):
            yield item

    def _resolve_lookup_ids_for_env(self, env, agent_id: str, reference_env=None) -> list[str]:
        lookup_ids = []

        def _append(candidate):
            cid = str(candidate or "").strip()
            if cid and cid not in lookup_ids:
                lookup_ids.append(cid)

        if reference_env is not None:
            _append(self._resolve_real_agent_id(reference_env, agent_id))
        if env is not None:
            _append(self._resolve_real_agent_id(env, agent_id))
        _append(agent_id)
        return lookup_ids

    def _collect_aircraft_view_snapshots(self, env, agent_id: str):
        reference_env = self._get_reference_env(env)
        snapshots = []
        for source_name, env_obj in self._iter_status_env_views(env):
            lookup_ids = self._resolve_lookup_ids_for_env(env_obj, agent_id, reference_env=reference_env)
            for collection_name in ('agents', '_jsbsims'):
                collection = getattr(env_obj, collection_name, None)
                if not isinstance(collection, dict):
                    continue
                matched = None
                matched_lookup_id = ""
                for lookup_id in lookup_ids:
                    aircraft = collection.get(lookup_id)
                    if aircraft is None:
                        continue
                    matched = aircraft
                    matched_lookup_id = lookup_id
                    break
                if matched is None:
                    continue
                snapshots.append({
                    "source": source_name,
                    "collection": collection_name,
                    "lookup_id": matched_lookup_id,
                    "aircraft": matched,
                    "alive": bool(getattr(matched, 'is_alive', False)),
                })
                break
        return snapshots

    def _get_status_source_priority(self, source_name: str) -> int:
        order = {
            "bridge_isolated": 0,
            "env_original": 1,
            "cap_sink": 2,
            "env": 3,
            "reference_original": 4,
            "reference": 5,
            "task_env": 6,
        }
        return int(order.get(str(source_name or "").strip(), 99))

    def _get_preferred_status_snapshots(self, snapshots):
        if not snapshots:
            return []
        normalized = list(snapshots)
        filtered = [
            snapshot for snapshot in normalized
            if str(snapshot.get("source")) != "bridge_isolated"
        ]
        if filtered:
            normalized = filtered
        authoritative_sources = {"env_original", "cap_sink", "env", "reference_original", "reference"}
        authoritative = [
            snapshot for snapshot in normalized
            if str(snapshot.get("source")) in authoritative_sources
        ]
        selected = authoritative if authoritative else normalized
        return sorted(
            selected,
            key=lambda snapshot: self._get_status_source_priority(snapshot.get("source")),
        )

    def _log_aircraft_view_conflict_if_needed(self, env, agent_id: str, snapshots) -> None:
        if not snapshots:
            return
        preferred_snapshots = self._get_preferred_status_snapshots(snapshots)
        if preferred_snapshots:
            snapshots = preferred_snapshots
        alive_values = {bool(snapshot.get("alive", False)) for snapshot in snapshots}
        if len(alive_values) <= 1:
            return
        current_step = int(getattr(env, 'current_step', -1)) if env is not None else -1
        if current_step < 0 or current_step % 80 != 0:
            return
        if not hasattr(self, '_aircraft_view_conflict_last'):
            self._aircraft_view_conflict_last = {}
        if self._aircraft_view_conflict_last.get(str(agent_id)) == current_step:
            return
        self._aircraft_view_conflict_last[str(agent_id)] = current_step
        detail = ";".join(
            f"{snapshot['source']}:{snapshot['collection']}:{snapshot['lookup_id']}={'Y' if snapshot['alive'] else 'N'}"
            for snapshot in snapshots
        )
        logging.warning(
            "[AIRCRAFT_VIEW_CONFLICT] %s step=%s views=%s",
            agent_id,
            current_step,
            detail or "-",
        )

    def _get_aircraft_from_any_collection(self, env, agent_id: str):
        snapshots = self._collect_aircraft_view_snapshots(env, agent_id)
        self._log_aircraft_view_conflict_if_needed(env, agent_id, snapshots)
        preferred_snapshots = self._get_preferred_status_snapshots(snapshots)
        if preferred_snapshots:
            snapshots = preferred_snapshots
        alive_snapshots = [snapshot for snapshot in snapshots if bool(snapshot.get("alive", False))]
        if alive_snapshots:
            return alive_snapshots[0]["aircraft"]
        if snapshots:
            return snapshots[0]["aircraft"]
        return None

    def _dedupe_friendly_members(self, members):
        deduped = []
        seen = set()
        for member in members or ():
            mid = str(member or "").strip()
            if not mid or mid in seen:
                continue
            seen.add(mid)
            deduped.append(mid)
        return self._sort_friendly_by_initial_lateral(deduped)

    def _get_nominal_friendly_pair_members(self, agent_id: str):
        if agent_id in ('A0100', 'A0200'):
            return ['A0100', 'A0200']
        if agent_id in ('A0300', 'A0400'):
            return ['A0300', 'A0400']
        return [agent_id]

    def _is_agent_alive_in_any_collection(self, env, agent_id: str) -> bool:
        snapshots = self._collect_aircraft_view_snapshots(env, agent_id)
        self._log_aircraft_view_conflict_if_needed(env, agent_id, snapshots)
        preferred_snapshots = self._get_preferred_status_snapshots(snapshots)
        if preferred_snapshots:
            snapshots = preferred_snapshots
        return any(bool(snapshot.get("alive", False)) for snapshot in snapshots)

    def _get_alive_formation_agents(self, env, agent_id: str, preserve_nominal_pair: bool = False):
        env_obj = self._get_reference_env(env if env is not None else getattr(self, 'env', None))
        alive_members = [
            aid for aid in self._get_formation_agents(agent_id, env_obj)
            if self._is_agent_alive_in_any_collection(env_obj, aid)
        ]
        if preserve_nominal_pair and str(agent_id).startswith('A') and len(alive_members) <= 1:
            nominal_members = [
                aid for aid in self._get_nominal_friendly_pair_members(agent_id)
                if self._is_agent_alive_in_any_collection(env_obj, aid)
            ]
            if agent_id in nominal_members and len(nominal_members) >= 2:
                alive_members = nominal_members
        return self._dedupe_friendly_members(alive_members)

    def _get_formation_agents(self, agent_id: str, env=None):
        env_obj = self._get_reference_env(env if env is not None else getattr(self, 'env', None))
        if agent_id.startswith('A') and env_obj is not None and self._is_bridge_isolated_env(env_obj):
            bridge_agent_id = str(agent_id)
            if hasattr(env_obj, 'resolve_virtual_agent_id'):
                try:
                    candidate = env_obj.resolve_virtual_agent_id(bridge_agent_id)
                    if candidate:
                        bridge_agent_id = str(candidate)
                except Exception:
                    pass
            visible_friendlies = self._dedupe_friendly_members(
                self._get_bridge_visible_alive_agent_ids(env_obj, 'A')
            )
            if bridge_agent_id in visible_friendlies:
                return visible_friendlies if len(visible_friendlies) >= 2 else [bridge_agent_id]
        if agent_id.startswith('A') and env_obj is not None:
            for group in self._resolve_friendly_formations(env_obj):
                if agent_id in group:
                    return self._dedupe_friendly_members(group)
        if agent_id in ('A0100', 'A0200'):
            return ['A0100', 'A0200']
        if agent_id in ('A0300', 'A0400'):
            return ['A0300', 'A0400']
        if agent_id in ('B0100', 'B0200'):
            return ['B0100', 'B0200']
        if agent_id in ('B0300', 'B0400'):
            return ['B0300', 'B0400']
        return [agent_id]

    def _get_formation_label(self, agent_id: str, env=None) -> str:
        if agent_id.startswith('A'):
            members = self._get_alive_formation_agents(env, agent_id, preserve_nominal_pair=True)
            if not members:
                members = self._get_formation_agents(agent_id, env)
            return self._make_friendly_group_label(tuple(members))
        members = self._get_formation_agents(agent_id, env)
        return '/'.join(members)

    def _get_teammate_id(self, agent_id: str, env=None):
        formation_agents = self._get_formation_agents(agent_id, env)
        for teammate_id in formation_agents:
            if teammate_id != agent_id:
                return teammate_id
        return None

    def _get_alive_team_agents(self, env, team_prefix: str) -> list[str]:
        canonical_ids = []
        if str(team_prefix).upper() == 'A':
            canonical_ids = ['A0100', 'A0200', 'A0300', 'A0400']
        elif str(team_prefix).upper() == 'B':
            canonical_ids = ['B0100', 'B0200', 'B0300', 'B0400']
        else:
            seen_agent_ids = set()
            for collection in self._iter_agent_collections(env):
                for other_id in collection.keys():
                    other_id = str(other_id)
                    if other_id.startswith(team_prefix) and other_id not in seen_agent_ids:
                        seen_agent_ids.add(other_id)
                        canonical_ids.append(other_id)
        return [aid for aid in canonical_ids if self._is_agent_alive_in_any_collection(env, aid)]

    def _get_best_armed_relief_snapshot(self, env, agent_id: str, target=None) -> dict:
        snapshot = {
            "agent_id": "",
            "missiles_left": 0,
            "same_formation": False,
            "depth_km": float('nan'),
            "target_distance_km": float('nan'),
            "depth_gap_km": float('nan'),
        }
        if not str(agent_id).startswith('A'):
            return snapshot

        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return snapshot

        my_depth_km = self._get_aircraft_battlefield_depth_km(env, aircraft)
        same_formation_members = set(self._get_formation_agents(agent_id, env))
        best_key = None

        for other_id in self._get_alive_team_agents(env, 'A'):
            if other_id == agent_id:
                continue
            other_aircraft = self._get_aircraft_from_any_collection(env, other_id)
            if other_aircraft is None or not getattr(other_aircraft, 'is_alive', False):
                continue

            missiles_left = self._get_aircraft_missiles_left(other_aircraft)
            if missiles_left <= 0:
                continue

            same_formation = other_id in same_formation_members
            other_depth_km = self._get_aircraft_battlefield_depth_km(env, other_aircraft)
            target_distance_km = float('nan')
            if target is not None and getattr(target, 'is_alive', False):
                try:
                    target_distance_km = float(self._calculate_distance_between(other_aircraft, target)) / 1000.0
                except Exception:
                    target_distance_km = float('nan')

            score_key = (
                1 if same_formation else 0,
                int(missiles_left),
                -float(target_distance_km) if np.isfinite(target_distance_km) else -9999.0,
                float(other_depth_km) if np.isfinite(other_depth_km) else -9999.0,
            )
            if best_key is not None and score_key <= best_key:
                continue

            best_key = score_key
            snapshot = {
                "agent_id": str(other_id),
                "missiles_left": int(missiles_left),
                "same_formation": bool(same_formation),
                "depth_km": float(other_depth_km),
                "target_distance_km": float(target_distance_km),
                "depth_gap_km": (
                    float(my_depth_km - other_depth_km)
                    if np.isfinite(my_depth_km) and np.isfinite(other_depth_km)
                    else float('nan')
                ),
            }

        return snapshot

    def _get_formation_role_by_position(self, env, agent_id: str) -> str:
        """按横向位置首判并锁定主僚机：首次判定后不再随位置变化而切换。"""
        formation = self._get_formation_agents(agent_id, env)
        formation_label = self._get_formation_label(agent_id, env)

        # 优先使用首次锁定结果（双机都存活时保持不变）
        role_lock = self._formation_role_lock.get(formation_label)
        if role_lock:
            locked_lead = role_lock.get('lead')
            locked_wingman = role_lock.get('wingman')
            lead_alive = bool(
                (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, locked_lead)
                )
            )
            wingman_alive = bool(
                (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, locked_wingman)
                )
            )

            if lead_alive and wingman_alive:
                return 'lead' if agent_id == locked_lead else 'wingman'

            # 仅剩单机时，退化为单机主机角色，避免后续决策链中断
            if lead_alive and not wingman_alive:
                return 'lead' if agent_id == locked_lead else 'wingman'
            if wingman_alive and not lead_alive:
                return 'lead' if agent_id == locked_wingman else 'wingman'

        alive_positions = []
        for aid in formation:
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is not None and getattr(aircraft, 'is_alive', False):
                try:
                    pos = aircraft.get_position()
                    alive_positions.append((aid, float(pos[1])))  # pos[1]=东向分量
                except Exception:
                    continue

        if len(alive_positions) >= 2:
            alive_positions.sort(key=lambda it: it[1])  # 东向值从小到大
            lead_id = alive_positions[0][0]  # 最西侧（左侧）= 长机
            wingman_id = alive_positions[1][0]

            # 首次判定后锁定，不再因机动交叉而交换
            if not role_lock:
                self._formation_role_lock[formation_label] = {
                    'lead': lead_id,
                    'wingman': wingman_id,
                }
            return 'lead' if agent_id == lead_id else 'wingman'

        default_leads = {'A0100', 'A0300', 'B0100', 'B0300'}
        return 'lead' if agent_id in default_leads else 'wingman'

    def _is_formation_lead(self, env, agent_id: str) -> bool:
        return self._get_formation_role_by_position(env, agent_id) == 'lead'

    def _get_formation_reset_manager(self, agent_id: str):
        formation_label = self._get_formation_label(agent_id, self._get_reference_env())
        if formation_label not in self._formation_reset_managers:
            from advanced_formation_reset_manager import AdvancedFormationResetManager
            self._formation_reset_managers[formation_label] = AdvancedFormationResetManager()
        return self._formation_reset_managers.get(formation_label, self.formation_reset_manager)

    def _get_formation_alive_count(self, env, agent_id: str) -> int:
        return sum(
            1
            for aid in self._get_formation_agents(agent_id, env)
            if (
                (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, aid)
                )
            )
        )

    def _get_min_distance_to_enemy_m(self, env, agent_id: str) -> float:
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return float('inf')
        distances = []
        for enemy_id in get_enemy_team(agent_id):
            enemy = self._get_aircraft_from_any_collection(env, enemy_id)
            if enemy is None or not getattr(enemy, 'is_alive', False):
                continue
            distances.append(self._calculate_distance_between(aircraft, enemy))
        return min(distances) if distances else float('inf')

    def _get_alive_enemy_count(self, env, agent_id: str) -> int:
        if self._is_bridge_isolated_env(env):
            return len(self._get_bridge_visible_alive_agent_ids(env, 'B'))
        return sum(1 for enemy_id in get_enemy_team(agent_id) if self._is_agent_alive_in_any_collection(env, enemy_id))

    def _count_alive_team_prefix(self, env, team_prefix: str) -> int:
        canonical_ids = []
        prefix = str(team_prefix or "").upper()
        if prefix == 'A':
            canonical_ids = ['A0100', 'A0200', 'A0300', 'A0400']
        elif prefix == 'B':
            canonical_ids = ['B0100', 'B0200', 'B0300', 'B0400']
        else:
            seen_agent_ids = set()
            for collection in self._iter_agent_collections(env):
                for other_id in collection.keys():
                    other_id = str(other_id)
                    if other_id.startswith(prefix) and other_id not in seen_agent_ids:
                        seen_agent_ids.add(other_id)
                        canonical_ids.append(other_id)
        return sum(1 for aid in canonical_ids if self._is_agent_alive_in_any_collection(env, aid))

    def _get_global_alive_enemy_count(self, env, agent_id: str) -> int:
        global_env = None
        if self._is_bridge_isolated_env(env):
            cap_task_sink = getattr(self, "_cap_task_sink", None)
            sink_env = getattr(cap_task_sink, "env", None) if cap_task_sink is not None else None
            original_env = getattr(env, "_original", None) if env is not None else None
            for candidate in (sink_env, original_env, getattr(sink_env, "_original", None) if sink_env is not None else None):
                if candidate is None or candidate is env:
                    continue
                global_env = candidate
                break
        if global_env is None:
            global_env = getattr(self, "env", None)
        if global_env is not None and global_env is not env:
            try:
                enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
                return int(self._count_alive_team_prefix(global_env, enemy_prefix))
            except Exception:
                pass
        try:
            enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
            return int(self._count_alive_team_prefix(env, enemy_prefix))
        except Exception:
            return -1

    def _get_aircraft_missiles_left(self, aircraft) -> int:
        if aircraft is None:
            return 0
        try:
            if hasattr(aircraft, 'num_left_missiles'):
                return max(0, int(getattr(aircraft, 'num_left_missiles')))
        except Exception:
            pass
        try:
            return max(0, int(getattr(aircraft, 'num_missiles', 0)))
        except Exception:
            return 0

    def _get_team_alive_count(self, env, team_prefix: str) -> int:
        if self._is_bridge_isolated_env(env):
            return len(self._get_bridge_visible_alive_agent_ids(env, team_prefix))
        seen_agent_ids = set()
        total = 0
        for collection in self._iter_agent_collections(env):
            for other_id, other_aircraft in collection.items():
                other_id = str(other_id)
                if other_id in seen_agent_ids or not other_id.startswith(team_prefix):
                    continue
                seen_agent_ids.add(other_id)
                if getattr(other_aircraft, 'is_alive', False):
                    total += 1
        return total

    def _get_bridge_visible_alive_agent_ids(self, env, team_prefix: str) -> list[str]:
        agents = getattr(env, 'agents', {}) or {}
        visible = []
        seen_real = set()
        prefix = str(team_prefix or "").upper()
        for visible_id, aircraft in agents.items():
            aid = str(visible_id or "").strip()
            if not aid.startswith(prefix):
                continue
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            real_id = str(getattr(aircraft, 'real_id', aid) or aid).strip()
            dedupe_key = real_id or aid
            if dedupe_key in seen_real:
                continue
            seen_real.add(dedupe_key)
            visible.append(aid)
        return self._sort_friendly_by_initial_lateral(visible)

    def _get_team_remaining_missiles(self, env, team_prefix: str) -> int:
        total = 0
        seen_agent_ids = set()
        for collection in self._iter_agent_collections(env):
            for other_id, other_aircraft in collection.items():
                other_id = str(other_id)
                if other_id in seen_agent_ids or not other_id.startswith(team_prefix):
                    continue
                seen_agent_ids.add(other_id)
                if not getattr(other_aircraft, 'is_alive', False):
                    continue
                total += self._get_aircraft_missiles_left(other_aircraft)
        return total

    def _get_aircraft_battlefield_position_km(self, env, aircraft) -> tuple[float, float]:
        if aircraft is None:
            return float('nan'), float('nan')

        owner_task = getattr(env, 'task', None)
        coord_sys = getattr(owner_task, 'coord_sys', None) if owner_task is not None else None
        if coord_sys is not None and hasattr(aircraft, 'get_geodetic'):
            try:
                geo = aircraft.get_geodetic()
                x_km, y_km = coord_sys.geodetic_to_battlefield(geo[0], geo[1])
                return float(x_km), float(y_km)
            except Exception:
                pass

        try:
            position = aircraft.get_position()
            return float(position[0]) / 1000.0, float(position[1]) / 1000.0
        except Exception:
            return float('nan'), float('nan')

    def _get_aircraft_ne_position_m(self, aircraft) -> tuple[float, float]:
        """Return aircraft position in the native N/E frame used by get_position()."""
        if aircraft is None:
            return float('nan'), float('nan')
        try:
            position = aircraft.get_position()
            return float(position[0]), float(position[1])
        except Exception:
            return float('nan'), float('nan')

    def _get_aircraft_battlefield_depth_km(self, env, aircraft) -> float:
        _, depth_km = self._get_aircraft_battlefield_position_km(env, aircraft)
        return float(depth_km)

    def _evaluate_red_defensive_counterattack_window(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        target_zone: str = "UNKNOWN",
        my_depth_km: float = float('nan'),
        target_depth_km: float = float('nan'),
        missiles_left: int = None,
        formation_missiles: int = None,
        guidance_commit: bool = False,
        forward_commit_state: dict = None,
    ) -> dict:
        result = {
            "eligible": False,
            "reason": "",
            "distance_km": float(distance) / 1000.0 if distance is not None and np.isfinite(distance) else float('nan'),
            "target_zone": str(target_zone or "UNKNOWN"),
            "my_depth_km": float(my_depth_km),
            "target_depth_km": float(target_depth_km),
        }
        if not str(agent_id).startswith('A'):
            return result

        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return result
        if target is None or not getattr(target, 'is_alive', False):
            _, target = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
            if target is None or not getattr(target, 'is_alive', False):
                _, target = self._get_nearest_alive_enemy_anywhere(env, agent_id)
        if target is None or not getattr(target, 'is_alive', False):
            return result

        if missiles_left is None:
            missiles_left = self._get_aircraft_missiles_left(aircraft)
        if formation_missiles is None:
            formation_missiles = self._get_formation_remaining_missiles(env, agent_id)
        if missiles_left <= 0 or formation_missiles <= 0:
            return result

        if distance is None or not np.isfinite(distance):
            try:
                distance = float(self._calculate_distance_between(aircraft, target))
            except Exception:
                distance = None
        if distance is None or not np.isfinite(distance):
            return result
        distance_km = float(distance) / 1000.0
        result["distance_km"] = distance_km

        if not np.isfinite(my_depth_km):
            my_depth_km = self._get_aircraft_battlefield_depth_km(env, aircraft)
        if not np.isfinite(target_depth_km):
            target_depth_km = self._get_aircraft_battlefield_depth_km(env, target)
        if not target_zone or str(target_zone).upper() == "UNKNOWN":
            try:
                target_zone = self._classify_target_risk_zone(target, env=env)
            except Exception:
                target_zone = "UNKNOWN"
        target_zone = str(target_zone or "UNKNOWN").upper()
        result["target_zone"] = target_zone
        result["my_depth_km"] = float(my_depth_km)
        result["target_depth_km"] = float(target_depth_km)

        if target_zone not in ("HIGH", "MEDIUM"):
            return result

        if target_zone == "HIGH":
            max_distance_km = 90.0
            max_my_depth_km = 110.0
            max_target_depth_km = 132.0
            retreat_depth_limit_km = 148.0
        else:
            max_distance_km = 72.0
            max_my_depth_km = 106.0
            max_target_depth_km = 168.0
            retreat_depth_limit_km = 155.0

        if distance_km > max_distance_km:
            return result
        if np.isfinite(my_depth_km) and my_depth_km > max_my_depth_km:
            return result
        if np.isfinite(target_depth_km) and target_depth_km > max_target_depth_km:
            return result

        target_uid = str(getattr(target, 'uid', '') or '')
        shooter_pair = self._get_formation_label(agent_id, env)
        if target_uid:
            if target_uid.endswith("0100") or target_uid.endswith("0200"):
                target_pair = "A0100/A0200"
                native_target_friendlies = ["A0100", "A0200"]
            elif target_uid.endswith("0300") or target_uid.endswith("0400"):
                target_pair = "A0300/A0400"
                native_target_friendlies = ["A0300", "A0400"]
            else:
                target_pair = ""
                native_target_friendlies = []
            if target_pair and shooter_pair and target_pair != shooter_pair:
                native_armed_exists = any(
                    (
                        (lambda other: other is not None and getattr(other, 'is_alive', False))(
                            self._get_aircraft_from_any_collection(env, fid)
                        )
                        and self._get_aircraft_missiles_left(self._get_aircraft_from_any_collection(env, fid)) > 0
                    )
                    for fid in native_target_friendlies
                )
                if native_armed_exists:
                    return result
                cross_pair_limit_km = 66.0 if target_zone == "HIGH" else 60.0
                if distance_km > cross_pair_limit_km:
                    return result

        commit_state = forward_commit_state if isinstance(forward_commit_state, dict) else {}
        enemy_wave_retreating = bool(commit_state.get("enemy_wave_retreating", False))
        closure_m_s = float(commit_state.get("closure_m_s", float('nan')))
        retreat_but_still_pressing = bool(
            (not enemy_wave_retreating)
            or target_zone == "HIGH"
            or distance_km <= 55.0
            or (np.isfinite(target_depth_km) and target_depth_km <= retreat_depth_limit_km)
            or (np.isfinite(closure_m_s) and closure_m_s >= -20.0)
            or bool(guidance_commit)
        )
        if not retreat_but_still_pressing:
            return result

        tags = ["defensive_counterfire", "high_zone" if target_zone == "HIGH" else "medium_zone"]
        tags.append("wave_retreat_close" if enemy_wave_retreating else "target_pressing")
        result["eligible"] = True
        result["reason"] = "|".join(tags)
        return result

    def _get_red_defensive_hold_members(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        reason: str = "",
    ) -> list[str]:
        formation_agents = [
            aid for aid in self._get_alive_formation_agents(
                env,
                agent_id,
                preserve_nominal_pair=True,
            )
            if str(aid).startswith('A')
        ] or ([agent_id] if str(agent_id).startswith('A') else [])

        hold_agents = []
        for aid in formation_agents:
            window = self._evaluate_red_defensive_counterattack_window(
                env,
                aid,
                target=target,
                distance=None if aid != agent_id else distance,
            )
            if window.get("eligible", False):
                continue
            hold_agents.append(aid)
        return hold_agents

    def _evaluate_red_defensive_posture(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        current_phase_name: str = "",
        reason: str = "",
    ) -> dict:
        result = {
            "preserve": False,
            "reason": "",
            "release_for_engage": False,
            "counterattack_window": False,
            "counterattack_reason": "",
            "target_zone": "UNKNOWN",
            "distance_km": float(distance) / 1000.0 if distance is not None and np.isfinite(distance) else float('nan'),
            "my_depth_km": float('nan'),
            "target_depth_km": float('nan'),
            "missiles_left": 0,
            "formation_missiles": 0,
            "team_missiles": 0,
            "formation_alive": 0,
            "friendly_alive": 0,
            "enemy_alive": 0,
            "guidance_commit": False,
            "enemy_intent": "UNKNOWN",
            "enemy_intent_source": "unknown",
            "closure_m_s": float('nan'),
            "retreat_heading_error_deg": float('nan'),
            "enemy_group_phase": "UNKNOWN",
            "enemy_group_id": "",
            "enemy_group_dynamic": False,
            "armed_relief_id": "",
            "armed_relief_missiles": 0,
            "armed_relief_same_formation": False,
            "armed_relief_depth_km": float('nan'),
            "armed_relief_target_distance_km": float('nan'),
            "armed_relief_gap_km": float('nan'),
        }
        if not str(agent_id).startswith('A'):
            return result

        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return result

        if target is None:
            _, target = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
            if target is None:
                _, target = self._get_nearest_alive_enemy_anywhere(env, agent_id)
        if target is not None and getattr(target, 'is_alive', False) and distance is None:
            try:
                distance = float(self._calculate_distance_between(aircraft, target))
            except Exception:
                distance = None

        try:
            target_zone = self._classify_target_risk_zone(target, env=env)
        except Exception:
            target_zone = "UNKNOWN"

        target_id = str(getattr(target, 'uid', '')) if target is not None else None
        guidance_commit = False
        try:
            guidance_commit = bool(self._formation_has_active_guidance_commit(env, agent_id, target_id=target_id))
        except Exception:
            guidance_commit = False

        missiles_left = self._get_aircraft_missiles_left(aircraft)
        formation_alive = self._get_formation_alive_count(env, agent_id)
        friendly_alive = self._get_team_alive_count(env, 'A')
        enemy_alive = self._get_alive_enemy_count(env, agent_id)
        formation_missiles = self._get_formation_remaining_missiles(env, agent_id)
        team_missiles = self._get_team_remaining_missiles(env, 'A')
        my_depth_km = self._get_aircraft_battlefield_depth_km(env, aircraft)
        target_depth_km = self._get_aircraft_battlefield_depth_km(env, target) if target is not None else float('nan')
        armed_relief = self._get_best_armed_relief_snapshot(env, agent_id, target=target)
        latch_key = self._get_formation_label(agent_id, env) if formation_alive >= 2 else agent_id
        if not hasattr(self, '_red_defensive_posture_latch'):
            self._red_defensive_posture_latch = {}
        latch_active = bool(self._red_defensive_posture_latch.get(latch_key, False))

        result.update({
            "target_zone": target_zone,
            "distance_km": float(distance) / 1000.0 if distance is not None and np.isfinite(distance) else float('nan'),
            "my_depth_km": float(my_depth_km),
            "target_depth_km": float(target_depth_km),
            "missiles_left": int(missiles_left),
            "formation_missiles": int(formation_missiles),
            "team_missiles": int(team_missiles),
            "formation_alive": int(formation_alive),
            "friendly_alive": int(friendly_alive),
            "enemy_alive": int(enemy_alive),
            "guidance_commit": bool(guidance_commit),
            "armed_relief_id": str(armed_relief.get("agent_id", "")),
            "armed_relief_missiles": int(armed_relief.get("missiles_left", 0) or 0),
            "armed_relief_same_formation": bool(armed_relief.get("same_formation", False)),
            "armed_relief_depth_km": float(armed_relief.get("depth_km", float('nan'))),
            "armed_relief_target_distance_km": float(armed_relief.get("target_distance_km", float('nan'))),
            "armed_relief_gap_km": float(armed_relief.get("depth_gap_km", float('nan'))),
        })

        forward_commit_state = self._evaluate_red_forward_commit_block(
            env,
            agent_id,
            target=target,
            distance=distance,
            current_phase_name=current_phase_name,
            guidance_commit=guidance_commit,
        )
        result.update({
            "enemy_intent": str(forward_commit_state.get("enemy_intent", "UNKNOWN")),
            "enemy_intent_source": str(forward_commit_state.get("enemy_intent_source", "unknown")),
            "closure_m_s": float(forward_commit_state.get("closure_m_s", float('nan'))),
            "retreat_heading_error_deg": float(forward_commit_state.get("retreat_heading_error_deg", float('nan'))),
            "enemy_group_phase": str(forward_commit_state.get("enemy_group_phase", "UNKNOWN")),
            "enemy_group_id": str(forward_commit_state.get("enemy_group_id", "")),
            "enemy_group_dynamic": bool(forward_commit_state.get("enemy_group_dynamic", False)),
            "enemy_wave_retreating": bool(forward_commit_state.get("enemy_wave_retreating", False)),
        })

        counterattack_window = self._evaluate_red_defensive_counterattack_window(
            env,
            agent_id,
            target=target,
            distance=distance,
            target_zone=target_zone,
            my_depth_km=my_depth_km,
            target_depth_km=target_depth_km,
            missiles_left=missiles_left,
            formation_missiles=formation_missiles,
            guidance_commit=guidance_commit,
            forward_commit_state=forward_commit_state,
        )
        result["release_for_engage"] = bool(counterattack_window.get("eligible", False))
        result["counterattack_window"] = bool(counterattack_window.get("eligible", False))
        result["counterattack_reason"] = str(counterattack_window.get("reason", "") or "")

        solo_or_broken = formation_alive <= 1
        winchester = missiles_left <= 0
        formation_winchester = formation_missiles <= 0
        outnumbered = enemy_alive > max(1, friendly_alive)
        team_thin = team_missiles <= max(1, enemy_alive)
        guard_hold_depth_km = 96.0
        boundary_guard = np.isfinite(my_depth_km) and my_depth_km >= 118.0
        boundary_guard_hard = np.isfinite(my_depth_km) and my_depth_km >= 138.0
        overextended = np.isfinite(my_depth_km) and my_depth_km >= 150.0
        deep_overextended = np.isfinite(my_depth_km) and my_depth_km >= 180.0
        low_zone_hold_line = (
            np.isfinite(my_depth_km)
            and my_depth_km >= (guard_hold_depth_km + 2.0)
            and target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN')
        )
        medium_zone_overpush = (
            np.isfinite(my_depth_km)
            and my_depth_km >= (guard_hold_depth_km + 14.0)
            and target_zone == 'MEDIUM'
            and (distance is None or not np.isfinite(distance) or distance >= 35000.0)
        )
        enemy_not_pressing = target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN')
        medium_buffer_only = target_zone == 'MEDIUM' and (distance is None or not np.isfinite(distance) or distance >= 45000.0)
        armed_relief_exists = bool(result.get("armed_relief_id", ""))
        armed_relief_gap_km = float(result.get("armed_relief_gap_km", float('nan')))
        armed_relief_target_distance_km = float(result.get("armed_relief_target_distance_km", float('nan')))
        armed_relief_can_take_over = bool(
            armed_relief_exists
            and not guidance_commit
            and (
                not np.isfinite(armed_relief_target_distance_km)
                or armed_relief_target_distance_km <= 120.0
                or target_zone != 'HIGH'
            )
        )
        front_winchester_handoff = bool(
            winchester
            and armed_relief_can_take_over
            and (
                bool(forward_commit_state.get("enemy_wave_retreating", False))
                or (distance is not None and np.isfinite(distance) and distance >= 28000.0)
                or (np.isfinite(armed_relief_gap_km) and armed_relief_gap_km >= 6.0)
                or target_zone in ('MEDIUM', 'LOW', 'OUTSIDE', 'UNKNOWN')
            )
        )
        pair_winchester = formation_alive >= 2 and formation_winchester and (
            target_zone != 'HIGH' or boundary_guard or outnumbered
        )
        pair_low_inventory = (
            formation_alive >= 2
            and formation_missiles <= max(0, formation_alive - 1)
            and not counterattack_window.get("eligible", False)
            and (
                target_zone != 'HIGH'
                or boundary_guard
                or outnumbered
            )
            and (distance is None or not np.isfinite(distance) or distance >= 32000.0)
        )
        release_latch = bool(
            counterattack_window.get("eligible", False)
            or (target_zone == 'HIGH' and not boundary_guard_hard)
            or (
                target_zone == 'MEDIUM'
                and distance is not None
                and np.isfinite(distance)
                and distance <= 26000.0
                and formation_missiles > 0
                and not boundary_guard_hard
            )
        )

        reasons = []
        if front_winchester_handoff:
            reasons.append("winchester_relief_handoff")
        if solo_or_broken and winchester:
            reasons.append("solo_winchester")
        if winchester and enemy_alive >= max(1, friendly_alive):
            reasons.append("winchester_guard")
        if pair_winchester:
            reasons.append("pair_winchester")
        if pair_low_inventory and team_thin:
            reasons.append("pair_low_inventory")
        if low_zone_hold_line:
            reasons.append("low_zone_hold_line")
        if medium_zone_overpush and (solo_or_broken or formation_winchester or team_thin or outnumbered):
            reasons.append("medium_zone_overpush")
        if boundary_guard and target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN'):
            reasons.append("low_zone_boundary")
        if boundary_guard and target_zone == 'MEDIUM' and (
            distance is None or not np.isfinite(distance) or distance >= 42000.0
        ):
            reasons.append("medium_zone_boundary")
        if boundary_guard and target_zone == 'HIGH' and (formation_winchester or solo_or_broken or outnumbered):
            reasons.append("high_zone_boundary")
        if enemy_not_pressing and (winchester or formation_winchester):
            reasons.append("winchester_low_zone")
        if solo_or_broken and outnumbered and target_zone in ('MEDIUM', 'HIGH'):
            reasons.append("solo_outnumbered")
        if enemy_not_pressing and solo_or_broken and (outnumbered or missiles_left <= 1):
            reasons.append("solo_low_zone")
        if overextended and target_zone != 'HIGH' and (winchester or solo_or_broken or team_thin):
            reasons.append("overextended_guard")
        if deep_overextended:
            reasons.append("deep_overextended")
        if medium_buffer_only and solo_or_broken and winchester:
            reasons.append("medium_buffer_hold")
        if forward_commit_state.get("block", False) and not counterattack_window.get("eligible", False):
            forward_reason = str(forward_commit_state.get("reason", "forward_commit_block"))
            if bool(forward_commit_state.get("enemy_wave_retreating", False)):
                reasons.insert(0, forward_reason)
            else:
                reasons.append(forward_reason)
        if latch_active and not release_latch and (
            formation_winchester or team_thin or solo_or_broken or overextended or boundary_guard
        ):
            reasons.append("latched_defense")

        if reasons:
            result["preserve"] = True
            result["reason"] = "|".join(reasons[:3])
        else:
            result["reason"] = reason or current_phase_name or ""
        if release_latch:
            self._red_defensive_posture_latch.pop(latch_key, None)
        elif result["preserve"] and (
            solo_or_broken
            or formation_winchester
            or team_thin
            or overextended
            or boundary_guard
            or low_zone_hold_line
            or medium_zone_overpush
        ):
            self._red_defensive_posture_latch[latch_key] = True
        elif not latch_active:
            self._red_defensive_posture_latch.pop(latch_key, None)
        return result

    def _get_red_enemy_commit_snapshot(self, env, agent_id: str, target=None, distance: float = None) -> dict:
        snapshot = {
            "enemy_intent": "UNKNOWN",
            "enemy_intent_source": "unknown",
            "closure_m_s": float('nan'),
            "retreat_heading_error_deg": float('nan'),
            "hot_heading_error_deg": float('nan'),
            "retreat_geometry": False,
            "opening": False,
            "enemy_group_phase": "UNKNOWN",
            "enemy_group_id": "",
            "enemy_group_dynamic": False,
            "enemy_group_pressure_level": 0,
            "enemy_group_pressure_tag": "",
            "enemy_group_wave_index": 0,
            "enemy_group_forward_limit_km": float('nan'),
            "enemy_group_pressing_override": False,
            "enemy_north_phase_reason": "",
            "target_zone": "UNKNOWN",
            "target_depth_km": float('nan'),
            "enemy_wave_retreating": False,
        }
        if not str(agent_id).startswith('A'):
            return snapshot

        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False) or target is None or not getattr(target, 'is_alive', False):
            return snapshot

        target_id = str(getattr(target, 'uid', '') or '')
        cap_group_snapshot = self._get_cap_enemy_group_snapshot(env, target_id) if target_id else {}
        if cap_group_snapshot:
            snapshot["enemy_group_phase"] = str(cap_group_snapshot.get("phase", "UNKNOWN"))
            snapshot["enemy_group_id"] = str(cap_group_snapshot.get("group_id", ""))
            snapshot["enemy_group_dynamic"] = bool(cap_group_snapshot.get("is_dynamic_group", False))
            snapshot["enemy_group_pressure_level"] = int(cap_group_snapshot.get("pressure_level", 0) or 0)
            snapshot["enemy_group_pressure_tag"] = str(cap_group_snapshot.get("pressure_tag", "") or "")
            snapshot["enemy_group_wave_index"] = int(cap_group_snapshot.get("wave_index", 0) or 0)
            snapshot["enemy_group_forward_limit_km"] = float(cap_group_snapshot.get("forward_limit_y_km", float('nan')))
        if hasattr(self, "_enemy_intent_cache") and target_id:
            try:
                cached = self._enemy_intent_cache.get(target_id)
            except Exception:
                cached = None
            if cached:
                cached_intent, cached_step = cached
                if cached_intent and (env.current_step - cached_step) <= 80:
                    snapshot["enemy_intent"] = str(cached_intent)
                    snapshot["enemy_intent_source"] = "cache"

        if snapshot["enemy_intent"] == "UNKNOWN" and getattr(self, "situation_algorithm_switcher", None) is not None:
            try:
                inferred_intent = self.situation_algorithm_switcher.recognize_intent(target, aircraft, env)
                if inferred_intent:
                    snapshot["enemy_intent"] = str(inferred_intent)
                    snapshot["enemy_intent_source"] = "algo"
            except Exception:
                pass

        try:
            my_pos = np.asarray(aircraft.get_position(), dtype=float)
            target_pos = np.asarray(target.get_position(), dtype=float)
            rel_pos = target_pos - my_pos
            rel_norm = float(np.linalg.norm(rel_pos))
        except Exception:
            rel_pos = None
            rel_norm = float('nan')

        if distance is None and np.isfinite(rel_norm):
            distance = rel_norm

        try:
            snapshot["target_depth_km"] = float(self._get_aircraft_battlefield_depth_km(env, target))
        except Exception:
            snapshot["target_depth_km"] = float('nan')
        try:
            snapshot["target_zone"] = self._classify_target_risk_zone(target, env=env)
        except Exception:
            snapshot["target_zone"] = "UNKNOWN"

        if rel_pos is not None and np.isfinite(rel_norm) and rel_norm > 1.0:
            try:
                my_vel = np.asarray(aircraft.get_velocity(), dtype=float)
                target_vel = np.asarray(target.get_velocity(), dtype=float)
                closure_m_s = float(-np.dot(rel_pos, (target_vel - my_vel)) / rel_norm)
                snapshot["closure_m_s"] = closure_m_s
                snapshot["opening"] = bool(closure_m_s <= -40.0)
            except Exception:
                pass

            try:
                bearing_to_friendly = float((np.degrees(np.arctan2(my_pos[1] - target_pos[1], my_pos[0] - target_pos[0])) + 360.0) % 360.0)
                target_heading = float(target.get_property_value(c.attitude_psi_deg))
                away_heading = float((bearing_to_friendly + 180.0) % 360.0)
                retreat_heading_error_deg = abs(self._normalize_angle_diff(target_heading - away_heading))
                hot_heading_error_deg = abs(self._normalize_angle_diff(target_heading - bearing_to_friendly))
                snapshot["retreat_heading_error_deg"] = retreat_heading_error_deg
                snapshot["hot_heading_error_deg"] = hot_heading_error_deg
                snapshot["retreat_geometry"] = bool(
                    retreat_heading_error_deg <= 55.0
                    and (
                        snapshot["opening"]
                        or hot_heading_error_deg >= 115.0
                        or (distance is not None and np.isfinite(distance) and distance <= 30000.0)
                    )
                )
                if snapshot["enemy_intent"] == "UNKNOWN" and snapshot["retreat_geometry"]:
                    snapshot["enemy_intent"] = "RETREAT"
                    snapshot["enemy_intent_source"] = "geometry"
            except Exception:
                pass

        north_phase_status = self._evaluate_cap_enemy_north_phase_status(
            env,
            target_id,
            target=target,
            distance=distance,
            target_zone=str(snapshot.get("target_zone", "UNKNOWN")),
            target_depth_km=float(snapshot.get("target_depth_km", float('nan'))),
        ) if target_id else {}
        if north_phase_status:
            snapshot["enemy_group_forward_limit_km"] = float(
                north_phase_status.get("forward_limit_y_km", snapshot.get("enemy_group_forward_limit_km", float('nan')))
            )
            snapshot["enemy_group_pressing_override"] = bool(north_phase_status.get("pressing_override", False))
            snapshot["enemy_north_phase_reason"] = str(north_phase_status.get("reason", "") or "")
            if bool(north_phase_status.get("disengaged", False)):
                snapshot["enemy_wave_retreating"] = True
                snapshot["enemy_intent"] = "RETREAT"
                snapshot["enemy_intent_source"] = "enemy_wave"
        else:
            if snapshot["enemy_group_phase"] == "REGROUP_NORTH":
                snapshot["enemy_wave_retreating"] = True
                snapshot["enemy_intent"] = "RETREAT"
                snapshot["enemy_intent_source"] = "enemy_wave"
            elif (
                snapshot["enemy_group_phase"] == "TURN_NORTH"
                and int(snapshot.get("enemy_group_pressure_level", 0) or 0) < 2
            ):
                snapshot["enemy_wave_retreating"] = True
                snapshot["enemy_intent"] = "RETREAT"
                snapshot["enemy_intent_source"] = "enemy_wave"

        return snapshot

    def _evaluate_red_forward_commit_block(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        current_phase_name: str = "",
        guidance_commit: bool = False,
    ) -> dict:
        result = {
            "block": False,
            "reason": "",
            "target_zone": "UNKNOWN",
            "my_depth_km": float('nan'),
            "target_depth_km": float('nan'),
            "enemy_intent": "UNKNOWN",
            "enemy_intent_source": "unknown",
            "closure_m_s": float('nan'),
            "retreat_heading_error_deg": float('nan'),
            "guidance_commit": bool(guidance_commit),
            "enemy_group_phase": "UNKNOWN",
            "enemy_group_id": "",
            "enemy_group_dynamic": False,
            "enemy_group_pressure_level": 0,
            "enemy_group_pressure_tag": "",
            "enemy_wave_retreating": False,
        }
        if not str(agent_id).startswith('A'):
            return result

        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return result

        if target is None:
            target_id = get_target_with_fallback(agent_id, env)
            target = self._get_aircraft_from_any_collection(env, target_id) if target_id else None
        if target is None or not getattr(target, 'is_alive', False):
            _, target = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
        if target is None or not getattr(target, 'is_alive', False):
            _, target = self._get_nearest_alive_enemy_anywhere(env, agent_id)
        if target is None or not getattr(target, 'is_alive', False):
            return result

        if distance is None:
            try:
                distance = float(self._calculate_distance_between(aircraft, target))
            except Exception:
                distance = None

        target_zone = self._classify_target_risk_zone(target, env=env)
        my_depth_km = self._get_aircraft_battlefield_depth_km(env, aircraft)
        target_depth_km = self._get_aircraft_battlefield_depth_km(env, target)
        formation_alive = self._get_formation_alive_count(env, agent_id)
        friendly_alive = self._get_team_alive_count(env, 'A')
        enemy_alive = self._get_alive_enemy_count(env, agent_id)
        missiles_left = self._get_aircraft_missiles_left(aircraft)
        formation_missiles = self._get_formation_remaining_missiles(env, agent_id)
        team_missiles = self._get_team_remaining_missiles(env, 'A')

        commit_snapshot = self._get_red_enemy_commit_snapshot(env, agent_id, target=target, distance=distance)
        enemy_intent = str(commit_snapshot.get("enemy_intent", "UNKNOWN"))
        enemy_group_phase = str(commit_snapshot.get("enemy_group_phase", "UNKNOWN"))
        enemy_wave_retreating = bool(commit_snapshot.get("enemy_wave_retreating", False))
        enemy_retreating = bool(
            enemy_intent in ("RETREAT", "ESCAPE", "DEFENSIVE")
            or commit_snapshot.get("retreat_geometry", False)
            or enemy_wave_retreating
        )
        outnumbered = enemy_alive > max(1, friendly_alive)
        solo_or_broken = formation_alive <= 1
        low_inventory = formation_missiles <= max(1, formation_alive)
        guard_line_crossed = np.isfinite(my_depth_km) and my_depth_km >= 104.0
        deep_guard_crossed = np.isfinite(my_depth_km) and my_depth_km >= 118.0
        medium_guard_crossed = np.isfinite(my_depth_km) and my_depth_km >= 124.0
        merge_window = bool(distance is not None and np.isfinite(distance) and distance <= 45000.0)
        close_merge_window = bool(distance is not None and np.isfinite(distance) and distance <= 35000.0)
        medium_or_lower_zone = target_zone in ("MEDIUM", "LOW", "OUTSIDE", "UNKNOWN")

        reasons = []
        if enemy_wave_retreating:
            reasons.append("enemy_regroup_phase_hold")
            if medium_or_lower_zone:
                reasons.append("enemy_regroup_north")
            elif close_merge_window:
                reasons.append("enemy_regroup_close_hold")
            if np.isfinite(my_depth_km) and my_depth_km >= 92.0:
                reasons.append("enemy_regroup_no_chase")
        if enemy_retreating and guard_line_crossed and medium_or_lower_zone:
            reasons.append("enemy_retreat_guard")
        if enemy_retreating and deep_guard_crossed and merge_window:
            reasons.append("enemy_retreat_close_break")
        if target_zone in ("LOW", "OUTSIDE") and np.isfinite(my_depth_km) and my_depth_km >= 98.0:
            reasons.append("low_zone_forward_commit")
        if target_zone == "MEDIUM" and medium_guard_crossed and (
            distance is None or not np.isfinite(distance) or distance >= 25000.0
        ):
            reasons.append("medium_zone_forward_commit")
        if close_merge_window and solo_or_broken and enemy_alive >= 2:
            reasons.append("solo_close_merge")
        if merge_window and (outnumbered or low_inventory or missiles_left <= 1 or team_missiles <= enemy_alive):
            reasons.append("close_merge_risk")
        if guidance_commit and enemy_retreating and deep_guard_crossed:
            reasons.append("guidance_commit_overrun")

        result.update({
            "block": bool(reasons),
            "reason": "|".join(reasons[:3]) if reasons else (current_phase_name or ""),
            "target_zone": target_zone,
            "my_depth_km": float(my_depth_km),
            "target_depth_km": float(target_depth_km),
            "enemy_intent": enemy_intent,
            "enemy_intent_source": str(commit_snapshot.get("enemy_intent_source", "unknown")),
            "closure_m_s": float(commit_snapshot.get("closure_m_s", float('nan'))),
            "retreat_heading_error_deg": float(commit_snapshot.get("retreat_heading_error_deg", float('nan'))),
            "enemy_group_phase": enemy_group_phase,
            "enemy_group_id": str(commit_snapshot.get("enemy_group_id", "")),
            "enemy_group_dynamic": bool(commit_snapshot.get("enemy_group_dynamic", False)),
            "enemy_group_pressure_level": int(commit_snapshot.get("enemy_group_pressure_level", 0) or 0),
            "enemy_group_pressure_tag": str(commit_snapshot.get("enemy_group_pressure_tag", "")),
            "enemy_wave_retreating": enemy_wave_retreating,
        })
        return result

    def _get_nearest_alive_enemy_in_current_env(self, env, agent_id: str):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return None, None

        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        best_enemy_id = None
        best_enemy = None
        min_distance = float('inf')

        for other_id, other_aircraft in getattr(env, 'agents', {}).items():
            other_id = str(other_id)
            if not other_id.startswith(enemy_prefix) or not getattr(other_aircraft, 'is_alive', False):
                continue
            try:
                distance = float(self._calculate_distance_between(aircraft, other_aircraft))
            except Exception:
                continue
            if distance < min_distance:
                min_distance = distance
                best_enemy_id = other_id
                best_enemy = other_aircraft

        return best_enemy_id, best_enemy

    def _get_nearest_alive_enemy_anywhere(self, env, agent_id: str):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return None, None

        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        best_enemy_id = None
        best_enemy = None
        min_distance = float('inf')

        for collection in self._iter_agent_collections_for_enemy_checks(env):
            for other_id, other_aircraft in collection.items():
                other_id = str(other_id)
                if not other_id.startswith(enemy_prefix) or not getattr(other_aircraft, 'is_alive', False):
                    continue
                try:
                    distance = float(self._calculate_distance_between(aircraft, other_aircraft))
                except Exception:
                    continue
                if distance < min_distance:
                    min_distance = distance
                    best_enemy_id = other_id
                    best_enemy = other_aircraft

        return best_enemy_id, best_enemy

    def _classify_target_risk_zone(self, target, env=None) -> str:
        if target is None:
            return "UNKNOWN"
        try:
            if env is not None:
                y_km = self._get_aircraft_battlefield_depth_km(env, target)
            else:
                position = target.get_position()
                y_km = float(position[1]) / 1000.0
        except Exception:
            return "UNKNOWN"
        if not np.isfinite(y_km):
            return "UNKNOWN"
        if y_km < 0.0:
            return "OUTSIDE"
        if y_km <= 100.0:
            return "HIGH"
        if y_km <= 200.0:
            return "MEDIUM"
        if y_km <= 300.0:
            return "LOW"
        return "OUTSIDE"

    def _has_alive_enemy_in_current_env(self, env, agent_id: str) -> bool:
        enemy_id, _ = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
        return enemy_id is not None

    def _get_formation_remaining_missiles(self, env, agent_id: str) -> int:
        total = 0
        for aid in self._get_formation_agents(agent_id, env):
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            try:
                missiles_left = (
                    int(getattr(aircraft, 'num_left_missiles'))
                    if hasattr(aircraft, 'num_left_missiles')
                    else int(getattr(aircraft, 'num_missiles', 0))
                )
            except Exception:
                missiles_left = int(getattr(aircraft, 'num_missiles', 0))
            total += max(0, missiles_left)
        return total

    def _get_formation_fired_missiles(self, env, agent_id: str) -> int:
        missile_counts = getattr(self, 'aircraft_missile_counts', {}) or {}
        total = 0
        for aid in self._get_formation_agents(agent_id, env):
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            total += max(0, int(missile_counts.get(aid, 0)))
        return total

    def _get_formation_spacing_m(self, env, agent_id: str) -> float:
        alive_members = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if (
                (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, aid)
                )
            )
        ]
        if len(alive_members) < 2:
            return 0.0
        aircraft_0 = self._get_aircraft_from_any_collection(env, alive_members[0])
        aircraft_1 = self._get_aircraft_from_any_collection(env, alive_members[1])
        if aircraft_0 is None or aircraft_1 is None:
            return 0.0
        return float(
            TacticalUtils.calculate_horizontal_distance(
                aircraft_0,
                aircraft_1,
            )
        )

    def _formation_has_incoming_missile(self, env, agent_id: str) -> bool:
        for aid in self._get_formation_agents(agent_id, env):
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            if any(getattr(missile, 'is_alive', False) for missile in getattr(aircraft, 'under_missiles', [])):
                return True
        return False

    def _should_force_direct_second_attack(self, env, agent_id: str) -> bool:
        if self._get_formation_alive_count(env, agent_id) < 2:
            return True

        min_distance_m = self._get_min_distance_to_enemy_m(env, agent_id)
        if not np.isfinite(min_distance_m):
            return False
        if min_distance_m <= 30000.0:
            return True
        incoming_missile = self._formation_has_incoming_missile(env, agent_id)
        if incoming_missile and min_distance_m <= 38000.0:
            return True

        formation_remaining_missiles = self._get_formation_remaining_missiles(env, agent_id)
        if formation_remaining_missiles <= 1 and min_distance_m <= 45000.0:
            return True

        formation_spacing_m = self._get_formation_spacing_m(env, agent_id)
        if formation_spacing_m > 22000.0 and min_distance_m <= 42000.0:
            return True

        return False

    def _should_use_template_tactic(self, env, agent_id: str, tactic: str) -> bool:
        template_tactics = {
            'FORMATION_RESET',
            'UNIFIED_SECOND_ATTACK',
            'DRAG_SHOOT',
            'PINCER_ATTACK',
            'HIGH_LOW_ATTACK',
            'FRONT_BACK',
            'SIDE_BY_SIDE',
        }
        if tactic not in template_tactics:
            return False
        if self._get_formation_alive_count(env, agent_id) < 2:
            return False
        if str(agent_id).startswith('A'):
            target_id = None
            target = None
            try:
                target_id = get_target_with_fallback(agent_id, env)
            except Exception:
                target_id = None
            if target_id:
                target = self._get_aircraft_from_any_collection(env, target_id)
            if target is None or not getattr(target, 'is_alive', False):
                _, target = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
            forward_commit_state = self._evaluate_red_forward_commit_block(
                env,
                agent_id,
                target=target,
                current_phase_name="template_gate",
            )
            if forward_commit_state.get("block", False):
                if env.current_step % 80 == 0:
                    logging.info(
                        "[前出约束] %s tactic=%s zone=%s my_depth=%.1fkm target_depth=%.1fkm "
                        "intent=%s/%s closure=%.1fm/s reason=%s",
                        agent_id,
                        tactic,
                        forward_commit_state.get("target_zone", "UNKNOWN"),
                        float(forward_commit_state.get("my_depth_km", float('nan'))),
                        float(forward_commit_state.get("target_depth_km", float('nan'))),
                        forward_commit_state.get("enemy_intent", "UNKNOWN"),
                        forward_commit_state.get("enemy_intent_source", "unknown"),
                        float(forward_commit_state.get("closure_m_s", float('nan'))),
                        forward_commit_state.get("reason", "forward_commit_block"),
                    )
                return False
        min_distance_m = self._get_min_distance_to_enemy_m(env, agent_id)
        min_template_distance_m = 60000.0
        if tactic == 'FRONT_BACK':
            min_template_distance_m = 50000.0
        if not np.isfinite(min_distance_m) or min_distance_m < min_template_distance_m:
            return False
        return not self._should_force_direct_second_attack(env, agent_id)

    def _should_bypass_template_tactic(self, env, agent_id: str, tactic: str) -> bool:
        template_tactics = {
            'FORMATION_RESET',
            'UNIFIED_SECOND_ATTACK',
            'DRAG_SHOOT',
            'PINCER_ATTACK',
            'HIGH_LOW_ATTACK',
            'FRONT_BACK',
            'SIDE_BY_SIDE',
        }
        if tactic not in template_tactics:
            return False
        return not self._should_use_template_tactic(env, agent_id, tactic)

    def _get_desired_salvo_size(self, env, agent_id: str, target_id: str, is_second_attack: bool = False) -> int:
        aircraft = env.agents.get(agent_id)
        target = env.agents.get(target_id) if target_id in env.agents else None
        if aircraft is None or target is None or not getattr(target, 'is_alive', False):
            return 0

        missiles_left = self._get_aircraft_missiles_left(aircraft)
        if missiles_left <= 0:
            return 0

        distance_km = float(self._calculate_distance_between(aircraft, target)) / 1000.0
        target_zone = self._classify_target_risk_zone(target, env=env)
        alive_enemy_count = sum(
            1
            for other_id, other_aircraft in env.agents.items()
            if str(other_id).startswith('B') and getattr(other_aircraft, 'is_alive', False)
        )
        active_on_target = 0
        missile_adapter = self._get_shared_missile_adapter(env)
        if missile_adapter is not None and hasattr(missile_adapter, '_missiles'):
            for status in getattr(missile_adapter, '_missiles', {}).values():
                try:
                    if (
                        getattr(status, 'target_id', None) == target_id
                        and str(getattr(status, 'guide_agent_id', '')).startswith('A')
                        and str(getattr(getattr(status, 'state', None), 'value', '')).lower() in ('launched', 'guiding', 'terminal')
                    ):
                        active_on_target += 1
                except Exception:
                    continue

        team_missiles_left = self._get_team_remaining_missiles(env, 'A')
        formation_size = self._get_formation_alive_count(env, agent_id)

        if missiles_left <= 1:
            return 1
        if active_on_target >= 1:
            return 1
        if team_missiles_left <= max(4, alive_enemy_count + 2):
            return 1
        if target_zone not in ('HIGH', 'MEDIUM') and (alive_enemy_count >= 2 or missiles_left <= 3):
            return 1
        if alive_enemy_count >= 3:
            return 1
        if is_second_attack:
            if formation_size < 2:
                return 1
            if (
                target_zone == 'HIGH'
                and alive_enemy_count == 1
                and distance_km <= 42.0
                and missiles_left >= 3
                and team_missiles_left >= 4
            ):
                return 2
            if (
                target_zone == 'MEDIUM'
                and alive_enemy_count == 1
                and distance_km <= 38.0
                and missiles_left >= 4
                and team_missiles_left >= 5
            ):
                return 2
            return 1
        if (
            target_zone == 'HIGH'
            and alive_enemy_count == 1
            and distance_km <= 40.0
            and missiles_left >= 3
            and team_missiles_left >= 4
        ):
            return 2
        return 1

    def _get_shared_missile_adapter(self, env=None):
        missile_adapter = getattr(self, 'missile_adapter', None)
        if missile_adapter is not None:
            return missile_adapter
        owner_task = getattr(env, 'task', None) if env is not None else None
        if owner_task is None and env is not None:
            original_env = getattr(env, '_original', None)
            owner_task = getattr(original_env, 'task', None) if original_env is not None else None
        if owner_task is not None:
            missile_adapter = getattr(owner_task, 'missile_adapter', None)
            if missile_adapter is not None:
                return missile_adapter
        return None

    def _get_agent_guided_missiles(self, agent_id: str, target_id: str = None, env=None):
        missile_adapter = self._get_shared_missile_adapter(env)
        if missile_adapter is None or not hasattr(missile_adapter, 'get_guided_missiles'):
            return []
        try:
            return list(missile_adapter.get_guided_missiles(guide_agent_id=agent_id, target_id=target_id))
        except Exception:
            return []

    def _has_active_guidance_commit(self, env, agent_id: str, target_id: str = None) -> bool:
        guided = self._get_agent_guided_missiles(agent_id, target_id=target_id, env=env)
        if not guided and target_id is not None:
            guided = self._get_agent_guided_missiles(agent_id, target_id=None, env=env)
        if not guided:
            return False
        try:
            current_time = float(env.current_step) * float(env.time_interval)
        except Exception:
            current_time = 0.0
        state_manager = getattr(self, 'state_manager', None)
        if state_manager is None:
            owner_task = getattr(env, 'task', None)
            state_manager = getattr(owner_task, 'state_manager', None) if owner_task is not None else None
        if state_manager is not None and hasattr(state_manager, 'is_in_guidance_protection'):
            try:
                if state_manager.is_in_guidance_protection(agent_id, current_time, protection_time=18.0):
                    return True
            except Exception:
                pass
        for status in guided:
            state_name = str(getattr(getattr(status, 'state', None), 'value', ''))
            if state_name == 'terminal':
                return True
            try:
                distance_to_target = float(getattr(status, 'distance_to_target', 999.0))
            except Exception:
                distance_to_target = 999.0
            try:
                time_of_flight = float(getattr(status, 'time_of_flight', 0.0))
            except Exception:
                time_of_flight = 0.0
            if time_of_flight <= 72.0:
                return True
            if np.isfinite(distance_to_target) and distance_to_target <= 65.0:
                return True
        return False

    def _formation_has_active_guidance_commit(self, env, agent_id: str, target_id: str = None) -> bool:
        formation_agents = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if str(aid).startswith('A')
            and aid in env.agents
            and getattr(env.agents[aid], 'is_alive', False)
        ]
        if not formation_agents:
            formation_agents = [agent_id] if str(agent_id).startswith('A') else []
        for aid in formation_agents:
            if self._has_active_guidance_commit(env, aid, target_id=target_id):
                return True
        return False

    def _mark_cap_defensive_engagement_hold(
        self,
        env,
        agent_ids,
        reason: str = "",
        hold_steps: int = 15,
    ) -> None:
        cap_task_sink = getattr(self, '_cap_task_sink', None)
        if cap_task_sink is None:
            return

        try:
            current_step = int(getattr(env, 'current_step', 0))
        except Exception:
            current_step = int(getattr(cap_task_sink, 'step_count', 0))
        suppress_until = current_step + max(1, int(hold_steps))

        suppress_store = getattr(cap_task_sink, '_defensive_engagement_suppressed_until', None)
        if not isinstance(suppress_store, dict):
            suppress_store = {}
            setattr(cap_task_sink, '_defensive_engagement_suppressed_until', suppress_store)

        alive_agents = []
        for aid in agent_ids:
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            alive_agents.append(str(aid))
            suppress_store[str(aid)] = max(int(suppress_store.get(str(aid), -1)), suppress_until)

        if not alive_agents:
            return

        radar = getattr(cap_task_sink, 'cap_radar', None)
        if radar is not None:
            for aid in alive_agents:
                try:
                    radar.lock_target(aid, None)
                except Exception:
                    pass

        coop_engagement = getattr(cap_task_sink, 'coop_engagement', None)
        impacted_targets = set()
        if coop_engagement is not None:
            assignments = getattr(coop_engagement, 'assignments', None)
            if isinstance(assignments, dict):
                for aid in alive_agents:
                    assignment = assignments.pop(aid, None)
                    target_id = getattr(assignment, 'target_id', None)
                    if target_id:
                        impacted_targets.add(str(target_id))

            tracking_status = getattr(coop_engagement, 'tracking_status', None)
            if isinstance(tracking_status, dict):
                for tid, status in list(tracking_status.items()):
                    primary = str(getattr(status, 'primary_tracker', '') or '')
                    secondary = str(getattr(status, 'secondary_tracker', '') or '')
                    if primary in alive_agents or secondary in alive_agents:
                        impacted_targets.add(str(tid))
                        tracking_status.pop(tid, None)

            last_assignment_time = getattr(coop_engagement, '_last_assignment_time', None)
            if isinstance(last_assignment_time, dict):
                for tid in impacted_targets:
                    last_assignment_time.pop(tid, None)

        trackers_store = getattr(cap_task_sink, '_stable_tracking_trackers', None)
        elapsed_store = getattr(cap_task_sink, '_stable_tracking_elapsed', None)
        last_ok_store = getattr(cap_task_sink, '_stable_tracking_last_ok', None)
        ready_store = getattr(cap_task_sink, '_stable_tracking_ready_announced', None)
        if isinstance(trackers_store, dict):
            for tid, trackers in list(trackers_store.items()):
                remaining = [tracker_id for tracker_id in trackers if tracker_id not in alive_agents]
                if len(remaining) >= 2:
                    trackers_store[tid] = remaining
                    continue
                trackers_store.pop(tid, None)
                if isinstance(elapsed_store, dict):
                    elapsed_store[tid] = 0.0
                if isinstance(last_ok_store, dict):
                    last_ok_store.pop(tid, None)
                if isinstance(ready_store, set):
                    ready_store.discard(tid)
        if impacted_targets:
            if isinstance(elapsed_store, dict):
                for tid in impacted_targets:
                    elapsed_store[tid] = 0.0
            if isinstance(last_ok_store, dict):
                for tid in impacted_targets:
                    last_ok_store.pop(tid, None)
            if isinstance(ready_store, set):
                for tid in impacted_targets:
                    ready_store.discard(tid)

        track_event_last = getattr(cap_task_sink, '_track_event_last', None)
        if isinstance(track_event_last, dict):
            for aid in alive_agents:
                track_event_last.pop(aid, None)

        if current_step % 80 == 0:
            logging.info(
                "🛡️ [CAP_DEFENSE_SUPPRESS] agents=%s hold=%.1fs reason=%s",
                "/".join(sorted(alive_agents)),
                max(1, int(hold_steps)) * float(getattr(env, 'time_interval', 0.2)),
                reason or 'defensive_guard',
            )

    def _execute_defensive_preserve_action(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        reason: str = "",
    ) -> tuple:
        plan = self._plan_red_defensive_guard_action(
            env,
            agent_id,
            target=target,
            distance=distance,
            reason=reason,
            source="_execute_defensive_preserve_action",
        )
        return tuple(plan.get("command") or (7, 8, 3))

    def _get_friendly_bvr_hard_floor_m(self, agent_id: str) -> float:
        return 35000.0 if str(agent_id).startswith('A') else 22000.0

    def _get_friendly_bvr_entry_floor_m(self, agent_id: str) -> float:
        return 40000.0 if str(agent_id).startswith('A') else 26000.0

    def _get_friendly_bvr_preempt_floor_m(self, agent_id: str) -> float:
        return 48000.0 if str(agent_id).startswith('A') else 30000.0

    def _get_friendly_bvr_escape_arming_m(self, agent_id: str) -> float:
        return 60000.0 if str(agent_id).startswith('A') else 34000.0

    def _get_friendly_bvr_escape_release_m(self, agent_id: str) -> float:
        return 52000.0 if str(agent_id).startswith('A') else 30000.0

    def _plan_red_defensive_guard_action(
        self,
        env,
        agent_id: str,
        target=None,
        distance: float = None,
        reason: str = "",
        source: str = "",
    ):
        preserve_reason = str(reason or 'defensive_guard')
        self._deescalate_red_formation_to_defense(env, agent_id, reason=preserve_reason)

        aircraft = None
        nearest_enemy_id = None
        nearest_enemy = None
        if str(agent_id).startswith('A'):
            aircraft = self._get_aircraft_from_any_collection(env, agent_id)
            if aircraft is not None and getattr(aircraft, 'is_alive', False):
                nearest_enemy_id, nearest_enemy = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                if target is None and nearest_enemy is not None:
                    target = nearest_enemy
                if distance is None and target is not None and getattr(target, 'is_alive', False):
                    try:
                        distance = float(self._calculate_distance_between(aircraft, target))
                    except Exception:
                        distance = None
                entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
                if distance is not None and np.isfinite(distance) and distance <= entry_floor_m and target is not None:
                    standoff_command = self._build_bvr_hard_escape_command(
                        env,
                        agent_id,
                        target,
                        distance,
                        reason=f"defensive_guard_entry_bvr_floor:{distance/1000.0:.1f}km|{preserve_reason}",
                    )
                    if standoff_command is not None:
                        if hasattr(self, '_log_key_event'):
                            self._log_key_event(
                                env,
                                agent_id,
                                "BVR底线接管",
                                current_time=float(getattr(env, 'current_step', 0)) * float(getattr(env, 'time_interval', 0.2)),
                                当前阶段=(self.state_manager.get_agent_phase(agent_id).value if hasattr(self, 'state_manager') else None),
                                当前战术=self._get_agent_tactic(agent_id),
                                进入函数=source or "_plan_red_defensive_guard_action",
                                执行机动="BVR_STANDOFF",
                                当前状态=f"owner=defensive_guard dist={distance/1000.0:.1f}km",
                                目标点=(str(getattr(target, 'uid', None)) if hasattr(target, 'uid') else None),
                                当前指令=str(standoff_command),
                                退出条件=f"distance > {entry_floor_m/1000.0:.0f}km / guidance_commit",
                                是否满足退出="否",
                            )
                        if env.current_step % 40 == 0:
                            logging.warning(
                                "[BVR底线约束] %s dist=%.1fkm reason=defensive_guard_entry_floor nearest=%s",
                                agent_id,
                                distance / 1000.0,
                                nearest_enemy_id or "-",
                            )
                        return {
                            "command": standoff_command,
                            "maneuver": "BVR_STANDOFF",
                            "state": f"owner=defensive_guard reason={preserve_reason}",
                        }

        hold_agents = self._get_red_defensive_hold_members(
            env,
            agent_id,
            target=target,
            distance=distance,
            reason=preserve_reason,
        )
        if hold_agents:
            self._mark_cap_defensive_engagement_hold(env, hold_agents, reason=preserve_reason)

        if self._get_formation_alive_count(env, agent_id) >= 2:
            defensive_corridor = self._build_multi_ship_defensive_hold_command(
                env,
                agent_id,
                target=target,
                distance=distance,
                reason=preserve_reason,
            )
            if defensive_corridor is not None:
                return {
                    "command": defensive_corridor,
                    "maneuver": "DEFENSIVE_CORRIDOR_HOLD_FINAL",
                    "state": f"owner=defensive_guard reason={preserve_reason}",
                }

        defensive_hold = self._build_solo_defensive_hold_command(
            env,
            agent_id,
            target=target,
            distance=distance,
            reason=preserve_reason,
        )
        if defensive_hold is not None:
            return {
                "command": defensive_hold,
                "maneuver": "DEFENSIVE_HOLD_FINAL",
                "state": f"owner=defensive_guard reason={preserve_reason}",
            }

        if target is not None and distance is not None and np.isfinite(distance):
            fallback_standoff = self._build_bvr_hard_escape_command(
                env,
                agent_id,
                target,
                distance,
                reason=f"defensive_guard_fallback:{preserve_reason}",
            )
            if fallback_standoff is not None:
                return {
                    "command": fallback_standoff,
                    "maneuver": "BVR_STANDOFF",
                    "state": f"owner=defensive_guard reason={preserve_reason}",
                }

        return {
            "command": (7, 8, 3),
            "maneuver": "DEFENSIVE_DEFAULT_FINAL",
            "state": f"owner=defensive_guard reason={preserve_reason}",
        }

    def _build_multi_ship_defensive_hold_command(self, env, agent_id: str, target=None, distance: float = None, reason: str = ""):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return None
        alive_members = self._get_alive_formation_agents(env, agent_id, preserve_nominal_pair=True)
        if len(alive_members) < 2:
            return None
        if (
            str(agent_id).startswith('A')
            and hasattr(self, '_build_cap_multi_ship_defensive_hold_command')
            and hasattr(self, '_resolve_maneuver_intent_command')
        ):
            current_time = float(getattr(env, 'current_step', 0)) * float(getattr(env, 'time_interval', 0.2))
            target_info = self._resolve_preserve_target(env, target=target) if hasattr(self, '_resolve_preserve_target') else {"track_id": "", "x_km": float("nan"), "y_km": float("nan"), "source": "none"}
            distance_km = self._normalize_preserve_distance_km(env, agent_id, target_info, distance=distance) if hasattr(self, '_normalize_preserve_distance_km') else (float(distance) / 1000.0 if distance is not None and np.isfinite(distance) and float(distance) > 1000.0 else float(distance) if distance is not None and np.isfinite(distance) else None)
            intent = self._build_cap_multi_ship_defensive_hold_command(
                env,
                agent_id,
                target_info,
                distance_km=distance_km,
                reason=reason,
                current_time=current_time,
            )
            if intent is not None:
                return self._resolve_maneuver_intent_command(env, agent_id, [intent], current_time=current_time)

        current_north_m, current_east_m = self._get_aircraft_ne_position_m(aircraft)
        current_north_km = float(current_north_m) / 1000.0
        current_east_km = float(current_east_m) / 1000.0
        reason_text = str(reason or "")
        target_zone = self._classify_target_risk_zone(target, env=env) if target is not None else "UNKNOWN"

        anchor_east_km = float(np.mean([
            float(getattr(self, '_friendly_initial_east_m', {}).get(aid, current_east_m)) / 1000.0
            for aid in alive_members
        ]))
        hold_depth_center_km = 116.0 if target_zone in ("LOW", "OUTSIDE", "UNKNOWN") else (112.0 if target_zone == "HIGH" else 122.0)
        no_chase_reason = any(
            tag in reason_text
            for tag in (
                "enemy_regroup_north",
                "enemy_regroup_no_chase",
                "winchester_relief_handoff",
                "low_zone_hold_line",
                "latched_defense",
                "pair_guard",
                "suppressed",
                "no_legal_target",
                "direct_motion_blocked",
                "bridge_motion_blocked",
                "bvr_floor_overrun",
            )
        )
        if any(tag in reason_text for tag in ("enemy_regroup_north", "enemy_regroup_no_chase", "winchester_relief_handoff", "low_zone_hold_line", "bvr_floor_overrun")):
            hold_depth_center_km = max(110.0, hold_depth_center_km)
        hold_depth_center_km = float(np.clip(hold_depth_center_km, 108.0, 128.0))

        no_chase_hold_depth_km = 96.0
        if any(tag in reason_text for tag in ("enemy_regroup", "winchester", "low_zone_hold_line", "latched_defense", "pair_guard", "suppressed", "no_legal_target")):
            no_chase_hold_depth_km = 92.0
        if any(tag in reason_text for tag in ("enemy_regroup_north", "enemy_regroup_no_chase", "low_zone_hold_line", "bvr_floor_overrun")):
            no_chase_hold_depth_km = 88.0
        no_chase_hold_depth_km = float(np.clip(no_chase_hold_depth_km, 84.0, 104.0))

        sorted_members = self._sort_friendly_by_initial_lateral(list(alive_members))
        lane_spacing_km = 8.0
        lane_offset_km = -lane_spacing_km / 2.0 if agent_id == sorted_members[0] else lane_spacing_km / 2.0
        hold_east_km = float(anchor_east_km + lane_offset_km)
        lane_error_km = float(current_east_km - hold_east_km)
        north_error_km = float(current_north_km - hold_depth_center_km)
        if not hasattr(self, '_multi_ship_defensive_hold_state'):
            self._multi_ship_defensive_hold_state = {}
        formation_label = self._get_formation_label(agent_id, env)
        corridor_state = self._multi_ship_defensive_hold_state.setdefault(formation_label, {})

        hold_profile = "no_chase" if no_chase_reason else f"zone:{target_zone}"
        captured_hold_depth_km = float(corridor_state.get("captured_hold_depth_km", float("nan")))
        if corridor_state.get("hold_profile") != hold_profile or not np.isfinite(captured_hold_depth_km):
            if no_chase_reason:
                captured_hold_depth_km = float(np.clip(current_north_km, 84.0, 112.0))
            else:
                captured_hold_depth_km = float(np.clip(hold_depth_center_km, 40.0, 128.0))
            corridor_state["captured_hold_depth_km"] = captured_hold_depth_km
            corridor_state["hold_profile"] = hold_profile
        if no_chase_reason:
            hold_depth_center_km = captured_hold_depth_km
            north_error_km = float(current_north_km - hold_depth_center_km)

        forced_south_recovery = bool(
            current_north_km >= (hold_depth_center_km + 6.0)
            or "bvr_floor_overrun" in reason_text
        )
        floor_overrun = "bvr_floor_overrun" in reason_text
        floor_distance_km = float(distance) / 1000.0 if distance is not None and np.isfinite(distance) else None
        entry_floor_km = float(self._get_friendly_bvr_entry_floor_m(agent_id)) / 1000.0
        release_floor_km = entry_floor_km + 5.0
        floor_escape_active = bool(corridor_state.get("floor_escape_active", False))
        if floor_overrun and floor_distance_km is not None:
            if floor_escape_active:
                floor_escape_active = floor_distance_km < release_floor_km
            else:
                floor_escape_active = floor_distance_km <= entry_floor_km
        else:
            floor_escape_active = False
        corridor_state["floor_escape_active"] = floor_escape_active
        if no_chase_reason and not forced_south_recovery and not floor_escape_active:
            current_heading = self._get_current_heading_deg(env, agent_id)
            if abs(north_error_km) >= 8.0:
                base_heading = 0.0 if north_error_km < 0.0 else 180.0
                correction_deg = float(np.clip(lane_error_km * 0.65, -14.0, 14.0))
                if base_heading < 90.0 or base_heading > 270.0:
                    target_heading = float((base_heading - correction_deg) % 360.0)
                else:
                    target_heading = float((base_heading + correction_deg) % 360.0)
                corridor_mode = "depth_rejoin"
            else:
                base_heading = 0.0 if north_error_km <= 0.0 else 180.0
                correction_deg = float(np.clip(lane_error_km * 1.2, -28.0, 28.0))
                if base_heading < 90.0 or base_heading > 270.0:
                    target_heading = float((base_heading - correction_deg) % 360.0)
                else:
                    target_heading = float((base_heading + correction_deg) % 360.0)
                hold_delta = abs(float(self._normalize_angle_diff(current_heading - base_heading)))
                if abs(lane_error_km) < 1.5 and abs(north_error_km) < 2.5 and hold_delta <= 18.0:
                    target_heading = float(current_heading % 360.0)
                    corridor_mode = "depth_hold"
                else:
                    corridor_mode = "lateral_rejoin"
        elif floor_escape_active:
            target_heading = float((180.0 + np.clip(lane_error_km * 0.25, -3.0, 3.0)) % 360.0)
            corridor_mode = "south_recovery"
        else:
            if abs(north_error_km) >= 8.0:
                base_heading = 0.0 if north_error_km < 0.0 else 180.0
                correction_deg = float(np.clip(lane_error_km * 1.3, -10.0, 10.0))
                if base_heading < 90.0 or base_heading > 270.0:
                    target_heading = float((base_heading - correction_deg) % 360.0)
                else:
                    target_heading = float((base_heading + correction_deg) % 360.0)
                corridor_mode = "depth_rejoin"
            else:
                leg_key = str(corridor_state.get(f"{agent_id}_leg", "") or "")
                if leg_key not in ("north", "south"):
                    leg_key = "south" if current_north_km > hold_depth_center_km else "north"
                if current_north_km >= hold_depth_center_km + 4.0:
                    leg_key = "south"
                elif current_north_km <= hold_depth_center_km - 8.0:
                    leg_key = "north"
                corridor_state[f"{agent_id}_leg"] = leg_key
                base_heading = 0.0 if leg_key == "north" else 180.0
                correction_deg = float(np.clip(lane_error_km * 0.8, -8.0, 8.0))
                if base_heading < 90.0 or base_heading > 270.0:
                    target_heading = float((base_heading - correction_deg) % 360.0)
                else:
                    target_heading = float((base_heading + correction_deg) % 360.0)
                corridor_mode = "corridor_hold"
        recovery_mode = bool(corridor_mode != "corridor_hold" or forced_south_recovery)
        lateral_rejoin_only = bool(corridor_mode in ("depth_rejoin", "lateral_rejoin") and abs(lane_error_km) >= 3.0)

        hdg_cmd = self._get_heading_cmd(env, agent_id, target_heading)
        alt_cmd = 7 if float(aircraft.get_property_value(c.position_h_sl_m)) >= 7500.0 else 8
        vel_cmd = 5 if (recovery_mode or abs(north_error_km) > 4.0 or abs(lane_error_km) > 3.0) else 4
        if floor_escape_active:
            vel_cmd = 6
        elif "bvr_floor_overrun" in reason_text:
            vel_cmd = max(vel_cmd, 6)

        if env.current_step % 40 == 0:
            self._log_key_event(
                env,
                agent_id,
                "多机回收走廊",
                current_time=float(getattr(env, 'current_step', 0)) * float(getattr(env, 'time_interval', 0.2)),
                当前阶段=(self.state_manager.get_agent_phase(agent_id).value if hasattr(self, 'state_manager') else None),
                当前战术=self._get_agent_tactic(agent_id),
                进入函数="_build_multi_ship_defensive_hold_command",
                执行机动="DEFENSIVE_CORRIDOR_HOLD",
                当前状态=f"pair={formation_label}, zone={target_zone}, lane={lane_error_km:.1f}km, north_err={north_error_km:.1f}km, recovery={recovery_mode}, south_recovery={forced_south_recovery}, lateral_rejoin={lateral_rejoin_only}, floor_escape={floor_escape_active}, mode={corridor_mode}, no_chase={no_chase_reason}, floor_dist={(f'{floor_distance_km:.1f}km' if floor_distance_km is not None else '-')}",
                目标点=f"hold(E={hold_east_km:.1f},N={hold_depth_center_km:.1f})km hdg={target_heading:.1f}",
                当前指令=str((alt_cmd, hdg_cmd, vel_cmd)),
                退出条件=f"reason_change / enemy_alive_change / corridor_capture, reason={reason_text or 'multi_ship_defensive_hold'}",
                是否满足退出="否",
            )

        return alt_cmd, hdg_cmd, vel_cmd

    def _execute_adaptive_attack(self, env, agent_id: str, current_time: float, reason: str = "") -> tuple:
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return (7, 8, 3)

        reason_text = str(reason or "").lower()
        if agent_id.startswith('A') and 'preserve' in reason_text:
            preserve_state = {"preserve": False}
            try:
                preserve_state = self._evaluate_red_defensive_posture(
                    env,
                    agent_id,
                    current_phase_name=reason or "defensive_guard",
                    reason=reason or "defensive_guard",
                )
            except Exception:
                preserve_state = {"preserve": False}
            current_tactic = str(self._get_agent_tactic(agent_id) or "").upper()
            release_for_engage = bool(preserve_state.get("release_for_engage", False))
            if preserve_state.get("preserve", False) or (current_tactic == 'DEFENSIVE_GUARD' and not release_for_engage):
                return self._execute_defensive_preserve_action(
                    env,
                    agent_id,
                    reason=preserve_state.get("reason", reason or "defensive_guard"),
                )

        original_target_id = get_target_with_fallback(agent_id, env)
        target_id = original_target_id
        target = self._get_aircraft_from_any_collection(env, target_id) if target_id else None
        if target is None or not getattr(target, 'is_alive', False):
            target_id, target = self._get_nearest_alive_enemy_in_current_env(env, agent_id)
            if target is not None and getattr(target, 'is_alive', False):
                if env.current_step % 120 == 0:
                    logging.warning(
                        f"[自适应攻击-目标重取] {agent_id} 原目标={original_target_id} 无效，"
                        f"改为当前env最近活目标 {target_id}"
                    )
            else:
                target_id, target = self._get_nearest_alive_enemy_anywhere(env, agent_id)
                if target is not None and getattr(target, 'is_alive', False):
                    if env.current_step % 120 == 0:
                        logging.warning(
                            f"[自适应攻击-跨域重取] {agent_id} 原目标={original_target_id} 无效，"
                            f"改为全局最近活目标 {target_id}"
                        )
                else:
                    target = None
                alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
                if target is not None and getattr(target, 'is_alive', False):
                    pass
                elif alive_enemy_exists:
                    regroup_command = self._build_dynamic_regroup_command(env, agent_id)
                    if regroup_command is not None:
                        if env.current_step % 120 == 0:
                            logging.warning(
                                f"[自适应攻击-无局部目标] {agent_id} 原目标={original_target_id}，"
                                f"当前桥接env无活目标但全局仍有敌机，改为编队重整等待重分配"
                            )
                        return regroup_command

                    if agent_id.startswith('A'):
                        defensive_patrol = self._build_red_defensive_patrol_command(
                            env,
                            agent_id,
                            target=None,
                            distance=None,
                            reason="no_local_target_patrol",
                        )
                        if defensive_patrol is not None:
                            if env.current_step % 120 == 0:
                                logging.info(
                                    f"[无局部目标-守区巡逻] {agent_id} 无局部目标，改为守区巡逻等待重分配"
                                )
                            return defensive_patrol
                    if agent_id.startswith('A') and self._get_formation_alive_count(env, agent_id) <= 1:
                        defensive_hold = self._build_solo_defensive_hold_command(
                            env,
                            agent_id,
                            target=None,
                            distance=None,
                            reason="solo_no_local_target",
                        )
                        if defensive_hold is not None:
                            return defensive_hold

                    current_heading = self._get_current_heading_deg(env, agent_id)
                    formation_axis_heading = current_heading
                    executor = getattr(self, 'executor', None)
                    if executor is not None and hasattr(executor, '_get_formation_axis_heading'):
                        try:
                            formation_axis_heading = float(executor._get_formation_axis_heading(env, agent_id, alpha=1.0))
                        except Exception:
                            formation_axis_heading = current_heading
                    current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
                    alt_cmd = 8 if current_altitude < 7500.0 else 7
                    hdg_cmd = self._get_heading_cmd(env, agent_id, formation_axis_heading)
                    if env.current_step % 120 == 0:
                        logging.warning(
                            f"[自适应攻击-无局部目标] {agent_id} 原目标={original_target_id}，"
                            f"当前桥接env无活目标但全局仍有敌机，保持编队轴向等待上层重分配"
                        )
                    return alt_cmd, hdg_cmd, 4

                return self.executor.execute_tactical_turn(env, agent_id)

        distance = float(self._calculate_distance_between(aircraft, target))
        target_zone = self._classify_target_risk_zone(target, env=env) if agent_id.startswith('A') else "UNKNOWN"
        missiles_left = self._get_aircraft_missiles_left(aircraft)
        formation_alive_count = self._get_formation_alive_count(env, agent_id)
        alive_enemy_count = self._get_alive_enemy_count(env, agent_id)
        is_second_attack = self.is_agent_second_attack(agent_id) if hasattr(self, 'is_agent_second_attack') else False
        try:
            current_phase = self.state_manager.get_agent_phase(agent_id)
            current_phase_name = getattr(current_phase, 'value', '')
        except Exception:
            current_phase_name = ''
        incoming_missiles = any(
            getattr(missile, 'is_alive', False)
            for missile in getattr(aircraft, 'under_missiles', [])
        )

        force_direct_second_attack = False
        if is_second_attack and formation_alive_count >= 2:
            try:
                force_direct_second_attack = bool(self._should_force_direct_second_attack(env, agent_id))
            except Exception:
                force_direct_second_attack = False
        formation_spacing_m = self._get_formation_spacing_m(env, agent_id) if formation_alive_count >= 2 else 0.0
        formation_fired_missiles = self._get_formation_fired_missiles(env, agent_id) if formation_alive_count >= 1 else 0
        severe_formation_split = bool(formation_alive_count >= 2 and formation_spacing_m > 22000.0)
        critical_formation_split = bool(formation_alive_count >= 2 and formation_spacing_m > 26000.0)
        close_commit_window = bool(
            agent_id.startswith('A')
            and is_second_attack
            and formation_alive_count >= 2
            and formation_spacing_m <= 18000.0
            and distance <= 62000.0
            and not force_direct_second_attack
        )
        guidance_commit = bool(
            agent_id.startswith('A')
            and self._formation_has_active_guidance_commit(env, agent_id)
        )
        prelaunch_hold_needed = bool(
            agent_id.startswith('A')
            and missiles_left > 0
            and formation_fired_missiles <= 0
            and not guidance_commit
            and not incoming_missiles
            and distance <= 78000.0
            and formation_spacing_m <= 16000.0
        )
        hard_bvr_floor_m = (
            float(self._get_friendly_bvr_preempt_floor_m(agent_id))
            if agent_id.startswith('A')
            else 22000.0
        )
        if agent_id.startswith('A') and distance <= hard_bvr_floor_m:
            standoff_command = self._build_bvr_hard_escape_command(
                env,
                agent_id,
                target,
                distance,
                reason=f"hard_bvr_floor:{distance/1000.0:.1f}km",
            )
            if standoff_command is not None:
                if env.current_step % 40 == 0:
                    logging.warning(
                        "[BVR底线约束] %s dist=%.1fkm zone=%s phase=%s reason=hard_bvr_floor",
                        agent_id,
                        distance / 1000.0,
                        target_zone,
                        current_phase_name or "-",
                    )
                return standoff_command
        if distance <= 18000.0 and incoming_missiles:
            return self.executor.execute_tactical_evasion(env, agent_id)

        defensive_state = self._evaluate_red_defensive_posture(
            env,
            agent_id,
            target=target,
            distance=distance,
            current_phase_name=current_phase_name,
            reason=reason,
        ) if agent_id.startswith('A') else {"preserve": False}
        if defensive_state.get("preserve", False):
            self._deescalate_red_formation_to_defense(
                env,
                agent_id,
                reason=defensive_state.get("reason", reason or current_phase_name or "defensive_guard"),
            )
            if env.current_step % 80 == 0:
                formation_label = self._get_formation_label(agent_id, env)
                logging.info(
                    "🛡️ [DEFENSIVE_LEASH] %s pair=%s zone=%s dist=%.1fkm my_depth=%.1fkm target_depth=%.1fkm "
                    "intent=%s/%s retreat=%d wave=%s grp=%s dyn=%d closure=%.1fm/s away=%.1fdeg guidance=%d msl=%d form_msl=%d team_msl=%d "
                    "relief=%s/%d gap=%.1fkm alive=%dv%d reason=%s",
                    agent_id,
                    formation_label,
                    defensive_state.get("target_zone", "UNKNOWN"),
                    distance / 1000.0,
                    float(defensive_state.get("my_depth_km", float('nan'))),
                    float(defensive_state.get("target_depth_km", float('nan'))),
                    defensive_state.get("enemy_intent", "UNKNOWN"),
                    defensive_state.get("enemy_intent_source", "unknown"),
                    int(defensive_state.get("enemy_wave_retreating", False)),
                    defensive_state.get("enemy_group_phase", "UNKNOWN"),
                    defensive_state.get("enemy_group_id", ""),
                    int(defensive_state.get("enemy_group_dynamic", False)),
                    float(defensive_state.get("closure_m_s", float('nan'))),
                    float(defensive_state.get("retreat_heading_error_deg", float('nan'))),
                    int(defensive_state.get("guidance_commit", False)),
                    int(defensive_state.get("missiles_left", 0)),
                    int(defensive_state.get("formation_missiles", 0)),
                    int(defensive_state.get("team_missiles", 0)),
                    defensive_state.get("armed_relief_id", ""),
                    int(defensive_state.get("armed_relief_missiles", 0)),
                    float(defensive_state.get("armed_relief_gap_km", float('nan'))),
                    int(defensive_state.get("friendly_alive", 0)),
                    int(defensive_state.get("enemy_alive", 0)),
                    defensive_state.get("reason", reason or current_phase_name or "defensive_guard"),
                )
                if any(tag in str(defensive_state.get("reason", "")) for tag in ("retreat", "forward_commit", "merge", "overrun", "regroup")):
                    logging.info(
                        "[前出约束] %s phase=%s zone=%s dist=%.1fkm my_depth=%.1fkm target_depth=%.1fkm "
                        "intent=%s/%s retreat=%d wave=%s grp=%s dyn=%d closure=%.1fm/s away=%.1fdeg guidance=%d reason=%s",
                        agent_id,
                        current_phase_name or "-",
                        defensive_state.get("target_zone", "UNKNOWN"),
                        distance / 1000.0,
                        float(defensive_state.get("my_depth_km", float('nan'))),
                        float(defensive_state.get("target_depth_km", float('nan'))),
                        defensive_state.get("enemy_intent", "UNKNOWN"),
                        defensive_state.get("enemy_intent_source", "unknown"),
                        int(defensive_state.get("enemy_wave_retreating", False)),
                        defensive_state.get("enemy_group_phase", "UNKNOWN"),
                        defensive_state.get("enemy_group_id", ""),
                        int(defensive_state.get("enemy_group_dynamic", False)),
                        float(defensive_state.get("closure_m_s", float('nan'))),
                        float(defensive_state.get("retreat_heading_error_deg", float('nan'))),
                        int(defensive_state.get("guidance_commit", False)),
                        defensive_state.get("reason", reason or current_phase_name or "defensive_guard"),
                    )
            plan = self._plan_red_defensive_guard_action(
                env,
                agent_id,
                target=target,
                distance=distance,
                reason=defensive_state.get("reason", reason or current_phase_name or "defensive_guard"),
                source="_execute_adaptive_attack.preserve",
            )
            return tuple(plan.get("command") or (7, 8, 3))

        if agent_id.startswith('A'):
            if not hasattr(self, '_adaptive_bvr_standoff_latch'):
                self._adaptive_bvr_standoff_latch = {}
            standoff_zone_suppressed = bool(
                target_zone == 'HIGH'
                or (target_zone == 'MEDIUM' and distance <= 60000.0)
            )
            standoff_candidate = (not guidance_commit) and (not close_commit_window) and (not standoff_zone_suppressed) and (
                missiles_left <= 0
                or (
                    (is_second_attack or current_phase_name in ('TR_DOR', 'DOR_DR', 'DR_MAR', 'BEYOND_MAR'))
                    and (formation_fired_missiles > 0 or incoming_missiles)
                )
            )
            standoff_entry_distance_m = 48000.0 if missiles_left > 0 else 64000.0
            standoff_exit_distance_m = standoff_entry_distance_m + 8000.0
            standoff_latched = bool(self._adaptive_bvr_standoff_latch.get(agent_id, False))
            if standoff_latched and severe_formation_split and distance >= 42000.0:
                standoff_latched = False
            use_standoff = False
            standoff_blocked_by_formation = bool(
                formation_alive_count >= 2
                and not close_commit_window
                and (
                    (formation_spacing_m > 18000.0 and distance >= 52000.0)
                    or (severe_formation_split and distance >= 42000.0)
                )
            )
            if standoff_candidate and not standoff_blocked_by_formation and not guidance_commit:
                if distance <= standoff_entry_distance_m:
                    use_standoff = True
                elif standoff_latched and distance <= standoff_exit_distance_m:
                    use_standoff = True
            self._adaptive_bvr_standoff_latch[agent_id] = use_standoff
            if guidance_commit and env.current_step % 80 == 0:
                logging.info(
                    "[机动仲裁] %s guidance_commit=1 hold_nose dist=%.1fkm spacing=%.1fkm",
                    agent_id,
                    distance / 1000.0,
                    formation_spacing_m / 1000.0,
                )
            elif standoff_blocked_by_formation and env.current_step % 80 == 0:
                logging.info(
                    "[机动仲裁] %s standoff_blocked split dist=%.1fkm spacing=%.1fkm",
                    agent_id,
                    distance / 1000.0,
                    formation_spacing_m / 1000.0,
                )
            elif standoff_zone_suppressed and env.current_step % 80 == 0:
                logging.info(
                    "[机动仲裁] %s standoff_blocked zone=%s dist=%.1fkm",
                    agent_id,
                    target_zone,
                    distance / 1000.0,
                )
            elif prelaunch_hold_needed and env.current_step % 80 == 0:
                logging.info(
                    "[机动仲裁] %s hold_nose_prelaunch dist=%.1fkm fired=%d spacing=%.1fkm",
                    agent_id,
                    distance / 1000.0,
                    formation_fired_missiles,
                    formation_spacing_m / 1000.0,
                )
            if use_standoff:
                if close_commit_window and env.current_step % 80 == 0:
                    logging.info(
                        "[机动仲裁] %s close_commit=1 keep_standoff dist=%.1fkm spacing=%.1fkm reason=%s",
                        agent_id,
                        distance / 1000.0,
                        formation_spacing_m / 1000.0,
                        reason or current_phase_name or 'adaptive',
                    )
                standoff_command = self._build_bvr_standoff_command(
                    env,
                    agent_id,
                    target,
                    distance,
                    reason=reason or current_phase_name or ('bvr_floor_latched' if standoff_latched else 'bvr_floor'),
                )
                if standoff_command is not None:
                    return standoff_command

        direct_attack_distance_m = 76000.0
        regroup_zone_suppressed = bool(
            agent_id.startswith('A')
            and (
                target_zone == 'HIGH'
                or (target_zone == 'MEDIUM' and distance <= 52000.0)
            )
        )
        if agent_id.startswith('A') and formation_alive_count >= 2:
            regroup_gate_distance_m = 68000.0 if (is_second_attack and not force_direct_second_attack) else 90000.0
        else:
            regroup_gate_distance_m = 76000.0 if (is_second_attack and formation_alive_count >= 2) else 95000.0

        if formation_alive_count >= 2 and (not close_commit_window) and (not guidance_commit) and (not regroup_zone_suppressed) and (
            distance >= regroup_gate_distance_m
            or (agent_id.startswith('A') and formation_spacing_m > 18000.0 and distance >= 62000.0)
            or (agent_id.startswith('A') and severe_formation_split and distance >= 42000.0)
            or (agent_id.startswith('A') and critical_formation_split and distance >= 36000.0)
            or (
                agent_id.startswith('A')
                and is_second_attack
                and not force_direct_second_attack
                and distance >= 56000.0
            )
        ) and (not prelaunch_hold_needed):
            regroup_command = self._build_dynamic_regroup_command(env, agent_id)
            if regroup_command is not None:
                return regroup_command

        launch_request_min_m = 42000.0 if agent_id.startswith('A') else 22000.0
        launch_request_max_m = 76000.0 if agent_id.startswith('A') else 80000.0
        if launch_request_min_m <= distance <= launch_request_max_m and missiles_left > 0:
            try:
                self.state_manager.missile_launched[agent_id] = True
            except Exception:
                pass
        elif close_commit_window and env.current_step % 80 == 0:
            logging.info(
                "🧭 [MANEUVER_ARBITRATION] %s close_commit=1 skip_regroup dist=%.1fkm spacing=%.1fkm",
                agent_id,
                distance / 1000.0,
                formation_spacing_m / 1000.0,
            )

        if (
            is_second_attack
            and formation_alive_count >= 2
            and distance >= 58000.0
            and not force_direct_second_attack
            and self._get_agent_tactic(agent_id) == 'ADAPTIVE_ATTACK'
        ):
            self._set_agent_tactic(agent_id, 'UNIFIED_SECOND_ATTACK')

        if agent_id.startswith('A') and distance <= 38000.0 and not incoming_missiles:
            standoff_command = self._build_bvr_standoff_command(
                env,
                agent_id,
                target,
                distance,
                reason=f"soft_bvr_floor:{distance/1000.0:.1f}km",
            )
            if standoff_command is not None:
                return standoff_command

        target_bearing = TacticalUtils.calculate_bearing(aircraft, target)
        current_heading = self._get_current_heading_deg(env, agent_id)
        nose_to_target_error = abs(self._normalize_angle_diff(target_bearing - current_heading))
        alive_members = self._get_alive_formation_agents(
            env,
            agent_id,
            preserve_nominal_pair=True,
        )
        offset_deg = 0.0
        if len(alive_members) >= 2:
            if agent_id == alive_members[0]:
                offset_deg = -12.0
            elif agent_id == alive_members[1]:
                offset_deg = 12.0
            if distance <= 50000.0:
                offset_deg *= 0.7
            elif distance <= direct_attack_distance_m:
                offset_deg *= 0.85
        if prelaunch_hold_needed:
            offset_deg *= 0.35

        force_nose_align = bool(
            agent_id.startswith('A')
            and not guidance_commit
            and (
                nose_to_target_error >= 65.0
                or (prelaunch_hold_needed and nose_to_target_error >= 42.0)
            )
        )
        if force_nose_align:
            offset_deg = 0.0

        desired_heading = float((target_bearing + offset_deg + 360.0) % 360.0)
        executor = getattr(self, 'executor', None)
        if (
            distance > direct_attack_distance_m
            and executor is not None
            and hasattr(executor, '_smooth_heading')
        ):
            if force_nose_align:
                desired_heading = float(target_bearing)
            else:
                heading_alpha = 0.22
                if agent_id.startswith('A') and prelaunch_hold_needed and nose_to_target_error >= 30.0:
                    heading_alpha = 0.48
                elif agent_id.startswith('A') and not guidance_commit and nose_to_target_error >= 45.0:
                    heading_alpha = 0.38
                desired_heading = float(
                    executor._smooth_heading(
                        f"adaptive:{agent_id}",
                        desired_heading,
                        alpha=heading_alpha,
                    )
                )
        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
        if current_altitude < 4500.0:
            alt_cmd = 10
        elif current_altitude < 6000.0:
            alt_cmd = 9
        elif current_altitude < 7500.0:
            alt_cmd = 8
        else:
            alt_cmd = 7
        if current_altitude < 5500.0 or distance > 55000.0:
            vel_cmd = 6
        elif distance > 35000.0:
            vel_cmd = 5
        else:
            vel_cmd = 4
        if distance <= direct_attack_distance_m and not (agent_id.startswith('A') and distance <= 50000.0):
            vel_cmd = max(vel_cmd, 5)
        hdg_cmd = self._get_heading_cmd(env, agent_id, desired_heading)

        if env.current_step % 100 == 0:
            reason_text = f" reason={reason}" if reason else ""
            logging.info(
                f"[自适应攻击] {agent_id} dist={distance/1000:.1f}km heading={desired_heading:.1f}° "
                f"formation={self._get_formation_label(agent_id, env)}{reason_text}"
            )
        if force_nose_align and env.current_step % 80 == 0:
            logging.info(
                "[自适应攻击-对准] %s align_to_target heading_err=%.1f dist=%.1fkm",
                agent_id,
                nose_to_target_error,
                distance / 1000.0,
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_bvr_standoff_command(
        self,
        env,
        agent_id: str,
        target,
        distance: float,
        reason: str = "",
        allow_guidance_commit: bool = False,
    ):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False) or target is None:
            return None
        if str(agent_id).startswith('A'):
            if self._formation_has_active_guidance_commit(env, agent_id) and not allow_guidance_commit:
                entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
                if distance is None or not np.isfinite(distance) or float(distance) > entry_floor_m:
                    return None

        target_bearing = TacticalUtils.calculate_bearing(aircraft, target)
        alive_members = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if (
                (lambda teammate: teammate is not None and getattr(teammate, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, aid)
                )
            )
        ]
        alive_members = self._sort_friendly_by_initial_lateral(alive_members)
        missiles_left = int(getattr(aircraft, 'num_left_missiles', getattr(aircraft, 'num_missiles', 0)))

        split_offset = 0.0
        if len(alive_members) >= 2:
            if distance <= 42000.0:
                base_split_offset = 12.0
            elif missiles_left > 0:
                base_split_offset = 15.0
            else:
                base_split_offset = 18.0
            if agent_id == alive_members[0]:
                split_offset = -base_split_offset
            elif agent_id == alive_members[1]:
                split_offset = base_split_offset
            formation_spacing_m = self._get_formation_spacing_m(env, agent_id)
            if formation_spacing_m > 15000.0:
                split_offset = 0.0
            elif formation_spacing_m > 12000.0:
                split_offset *= 0.25
            elif formation_spacing_m > 9000.0:
                split_offset *= 0.5

        desired_heading = float((target_bearing + 180.0 + split_offset + 360.0) % 360.0)
        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
        alt_cmd = 8 if current_altitude < 7500.0 else 7
        if distance <= 28000.0:
            vel_cmd = 6
        elif distance <= 52000.0:
            vel_cmd = 5
        else:
            vel_cmd = 5 if missiles_left > 0 else 4
        hdg_cmd = self._get_heading_cmd(env, agent_id, desired_heading)

        if env.current_step % 80 == 0:
            reason_text = f" reason={reason}" if reason else ""
            formation_label = self._get_formation_label(agent_id, env)
            logging.warning(
                f"[BVR牵制] {agent_id} pair={formation_label} dist={distance/1000:.1f}km heading={desired_heading:.1f}°"
                f"{reason_text}"
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_bvr_hard_escape_command(self, env, agent_id: str, target, distance: float, reason: str = ""):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False) or target is None:
            return None

        floor_reason = "|".join(
            part for part in (str(reason or ""), "bvr_floor_overrun") if part
        )
        current_north, current_east = self._get_aircraft_ne_position_m(aircraft)
        away_heading = float((TacticalUtils.calculate_bearing(aircraft, target) + 180.0) % 360.0)
        formation_agents = [
            aid for aid in self._get_alive_formation_agents(env, agent_id, preserve_nominal_pair=True)
            if str(aid).startswith('A')
        ] if str(agent_id).startswith('A') else [agent_id]
        if not formation_agents:
            formation_agents = [agent_id]
        sorted_members = self._sort_friendly_by_initial_lateral(list(formation_agents))
        anchor_samples = []
        initial_east_store = getattr(self, '_friendly_initial_east_m', {})
        for aid in sorted_members:
            sample = initial_east_store.get(aid, None)
            if sample is not None and np.isfinite(float(sample)):
                anchor_samples.append(float(sample))
        if not anchor_samples:
            anchor_samples.append(float(current_east))
        anchor_east = float(np.mean(anchor_samples))
        lane_offset = 0.0
        if len(sorted_members) >= 2 and agent_id in sorted_members[:2]:
            lane_offset = -4000.0 if agent_id == sorted_members[0] else 4000.0
        escape_lane_east = float(anchor_east + lane_offset)
        lateral_error = float(current_east - escape_lane_east)
        south_heading = float((180.0 - np.clip(lateral_error / 450.0, -12.0, 12.0)) % 360.0)
        away_delta = float(self._normalize_angle_diff(away_heading - south_heading))
        desired_heading = float((south_heading + np.clip(away_delta, -18.0, 18.0)) % 360.0)

        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
        alt_cmd = 8 if current_altitude < 7500.0 else 7
        entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
        hard_floor_m = float(self._get_friendly_bvr_hard_floor_m(agent_id))
        vel_cmd = 6 if distance <= (entry_floor_m + 6000.0) else 5
        if distance <= hard_floor_m:
            vel_cmd = 6
        hdg_cmd = self._get_heading_cmd(env, agent_id, desired_heading)

        if env.current_step % 40 == 0:
            reason_text = f" reason={floor_reason}" if floor_reason else ""
            formation_label = self._get_formation_label(agent_id, env)
            logging.warning(
                f"[BVR硬逃逸] {agent_id} pair={formation_label} dist={distance/1000:.1f}km "
                f"mode=direct_escape heading={desired_heading:.1f}° south={south_heading:.1f}° "
                f"away={away_heading:.1f}° lane_err={lateral_error/1000.0:.1f}km"
                f"{reason_text}"
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_dynamic_regroup_command(self, env, agent_id: str):
        formation_agents = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if (
                (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, aid)
                )
            )
        ]
        if len(formation_agents) < 2:
            return None
        if str(agent_id).startswith('A') and self._formation_has_active_guidance_commit(env, agent_id):
            return None

        formation_agents = self._sort_friendly_by_initial_lateral(formation_agents)
        lead_id, wingman_id = formation_agents[0], formation_agents[1]
        lead_aircraft = self._get_aircraft_from_any_collection(env, lead_id)
        wing_aircraft = self._get_aircraft_from_any_collection(env, wingman_id)
        if lead_aircraft is None or wing_aircraft is None:
            return None
        lead_pos = lead_aircraft.get_position()
        wing_pos = wing_aircraft.get_position()
        north_values = {
            lead_id: float(lead_pos[0]),
            wingman_id: float(wing_pos[0]),
        }
        east_values = {
            lead_id: float(lead_pos[1]),
            wingman_id: float(wing_pos[1]),
        }
        formation_distance = float(np.hypot(
            north_values[lead_id] - north_values[wingman_id],
            east_values[lead_id] - east_values[wingman_id],
        ))
        longitudinal_spread = float(abs(north_values[lead_id] - north_values[wingman_id]))
        current_spacing = float(abs(east_values[lead_id] - east_values[wingman_id]))
        if formation_distance <= 11000.0 and longitudinal_spread <= 3500.0 and abs(current_spacing - 9260.0) <= 2200.0:
            return None
        my_aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if my_aircraft is None:
            return None

        target_distance = float("inf")
        target_zone = "UNKNOWN"
        try:
            regroup_target_id = get_target_with_fallback(agent_id, env)
            regroup_target = self._get_aircraft_from_any_collection(env, regroup_target_id)
            if regroup_target is not None and getattr(regroup_target, 'is_alive', False):
                target_distance = TacticalUtils.calculate_distance_between(my_aircraft, regroup_target)
                target_zone = self._classify_target_risk_zone(regroup_target, env=env)
        except Exception:
            target_distance = float("inf")
            target_zone = "UNKNOWN"
        if (
            str(agent_id).startswith('A')
            and target_zone in ('HIGH', 'MEDIUM')
            and target_distance <= 52000.0
        ):
            return None
        if (
            target_distance <= 70000.0
            and formation_distance <= 18000.0
            and longitudinal_spread <= 12000.0
        ):
            return None

        anchor_samples = []
        initial_east_store = getattr(self, '_friendly_initial_east_m', {})
        for aid in formation_agents:
            initial_east = initial_east_store.get(aid, None)
            if initial_east is not None and np.isfinite(float(initial_east)):
                anchor_samples.append(float(initial_east))
                continue
            aircraft = self._get_aircraft_from_any_collection(env, aid)
            if aircraft is None or not getattr(aircraft, 'is_alive', False):
                continue
            try:
                anchor_samples.append(float(aircraft.get_position()[1]))
            except Exception:
                continue
        if not anchor_samples:
            return None
        anchor_east = float(np.mean(anchor_samples))
        try:
            regroup_target_id = get_target_with_fallback(agent_id, env)
            regroup_target = self._get_aircraft_from_any_collection(env, regroup_target_id)
            if regroup_target is not None and getattr(regroup_target, 'is_alive', False):
                target_east = float(regroup_target.get_position()[1])
                anchor_east = float(np.clip(0.65 * anchor_east + 0.35 * target_east, anchor_east - 18000.0, anchor_east + 18000.0))
        except Exception:
            pass
        rear_north = float(min(north_values.values()))
        regroup_push_north = float(np.clip(longitudinal_spread * 0.25, 5000.0, 12000.0))
        anchor_north = rear_north + regroup_push_north
        spacing_m = 9260.0
        slot_north = {
            lead_id: anchor_north + 600.0,
            wingman_id: anchor_north - 600.0,
        }
        slot_east = {
            lead_id: anchor_east - spacing_m / 2.0,
            wingman_id: anchor_east + spacing_m / 2.0,
        }

        my_pos = my_aircraft.get_position()
        target_north = slot_north[agent_id]
        target_east = slot_east[agent_id]
        slot_error = float(np.hypot(float(my_pos[0]) - target_north, float(my_pos[1]) - target_east))
        if slot_error <= 1800.0 and longitudinal_spread <= 3500.0:
            return None

        north_error = float(my_pos[0]) - target_north
        east_error = float(my_pos[1]) - target_east
        target_heading = self._heading_to_point(env, agent_id, target_north, target_east)
        if north_error > 2500.0 and abs(east_error) < 12000.0:
            if agent_id == lead_id:
                target_heading = 105.0
            else:
                target_heading = 255.0
        current_altitude = float(my_aircraft.get_property_value(c.position_h_sl_m))
        alt_cmd = 8 if current_altitude < 7500.0 else 7
        if north_error > 2500.0:
            vel_cmd = 3
        elif north_error < -1500.0 or slot_error > 12000.0:
            vel_cmd = 6
        else:
            vel_cmd = 5
        hdg_cmd = self._get_heading_cmd(env, agent_id, target_heading)
        if env.current_step % 100 == 0:
            logging.info(
                f"[编队恢复] {agent_id} slot_error={slot_error/1000:.1f}km spread={longitudinal_spread/1000:.1f}km "
                f"target=({target_north/1000:.1f},{target_east/1000:.1f})km pair={self._get_formation_label(agent_id, env)}"
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_red_defensive_patrol_command(self, env, agent_id: str, target=None, distance: float = None, reason: str = ""):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return None
        if not str(agent_id).startswith('A'):
            return None

        alive_members = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if (
                (lambda teammate: teammate is not None and getattr(teammate, 'is_alive', False))(
                    self._get_aircraft_from_any_collection(env, aid)
                )
            )
        ]
        alive_members = self._sort_friendly_by_initial_lateral(alive_members)
        if not alive_members:
            return None

        # A defensive patrol command is a hard preserve state. Refresh the CAP-side
        # engagement hold here so coop-track / launch readiness cannot be rebuilt
        # while the tactical layer is already ordering no-chase patrol.
        hold_members = self._get_red_defensive_hold_members(
            env,
            agent_id,
            target=target,
            distance=distance,
            reason=reason,
        )
        if hold_members:
            self._mark_cap_defensive_engagement_hold(
                env,
                hold_members,
                reason=reason or "defensive_patrol",
                hold_steps=30,
            )

        formation_label = self._get_formation_label(agent_id, env)
        if not hasattr(self, '_red_defensive_patrol_state'):
            self._red_defensive_patrol_state = {}
        patrol_state = self._red_defensive_patrol_state.setdefault(
            formation_label,
            {"leg_sign": 1, "last_switch_time": -1e9, "pair_recovery_mode": False},
        )
        current_time_s = float(getattr(env, 'current_step', 0) * getattr(env, 'time_interval', 0.2))
        recreate_state = {}
        recreate_store = getattr(self, '_friendly_sim_recreate_state', None)
        if isinstance(recreate_store, dict):
            recreate_state = recreate_store.get(agent_id, {}) or {}
        recreate_posture_hold = current_time_s < float(recreate_state.get("posture_hold_until_s", -1e9) or -1e9)
        recreate_recovering = current_time_s < float(recreate_state.get("recover_until_s", -1e9) or -1e9)

        my_north_m, my_east_m = self._get_aircraft_ne_position_m(aircraft)
        my_north_km = float(my_north_m) / 1000.0
        my_east_km = float(my_east_m) / 1000.0
        if not np.isfinite(my_north_km) or not np.isfinite(my_east_km):
            return None

        target_zone = self._classify_target_risk_zone(target, env=env) if target is not None else "UNKNOWN"
        missiles_left = self._get_aircraft_missiles_left(aircraft)
        formation_missiles = self._get_formation_remaining_missiles(env, agent_id)
        team_missiles = self._get_team_remaining_missiles(env, 'A')
        friendly_alive = self._get_team_alive_count(env, 'A')
        enemy_alive = self._get_global_alive_enemy_count(env, agent_id)

        reason_text = str(reason or "")
        requested_no_enemy_guard = "mission_guard_no_enemy" in reason_text
        mission_guard_no_enemy = bool(enemy_alive <= 0)
        if requested_no_enemy_guard and enemy_alive > 0 and env.current_step % 60 == 0:
            logging.warning(
                "[DEFENSIVE_PATROL_REASON_MISMATCH] %s pair=%s requested=mission_guard_no_enemy enemy_alive=%d reason=%s",
                agent_id,
                formation_label,
                int(enemy_alive),
                reason_text or "-",
            )
        aggressive_breakaway = any(
            tag in reason_text
            for tag in ("close_merge", "forward_commit", "enemy_retreat", "guidance_commit_overrun")
        )
        winchester_relief_handoff = "winchester_relief_handoff" in reason_text

        hold_depth_center_km = 96.0
        if target_zone == 'HIGH':
            hold_depth_center_km = 88.0
        elif target_zone == 'MEDIUM':
            hold_depth_center_km = 102.0 if (formation_missiles <= 0 or missiles_left <= 0) else 98.0
        elif target_zone in ('LOW', 'OUTSIDE', 'UNKNOWN'):
            hold_depth_center_km = 92.0 if team_missiles <= 0 else 96.0
        if mission_guard_no_enemy:
            hold_depth_center_km = 92.0
        if enemy_alive > friendly_alive:
            hold_depth_center_km += 4.0
        if aggressive_breakaway:
            hold_depth_center_km = min(hold_depth_center_km, 88.0 if len(alive_members) >= 2 else 84.0)
        if winchester_relief_handoff:
            hold_depth_center_km = min(hold_depth_center_km, 92.0 if len(alive_members) >= 2 else 90.0)
        if any(tag in reason_text for tag in ("enemy_regroup_north", "enemy_regroup_no_chase", "low_zone_hold_line")):
            hold_depth_center_km = min(hold_depth_center_km, 88.0 if len(alive_members) >= 2 else 84.0)
        hold_depth_center_km = float(np.clip(hold_depth_center_km, 84.0, 110.0))

        solo_hard_recovery = bool(
            len(alive_members) <= 1
            and my_north_km >= (hold_depth_center_km + 18.0)
            and (
                aggressive_breakaway
                or missiles_left <= 0
                or formation_missiles <= 0
                or enemy_alive >= max(2, friendly_alive)
                or any(tag in reason_text for tag in ("overextended", "latched_defense", "enemy_regroup", "enemy_retreat"))
            )
        )
        if solo_hard_recovery:
            # A solo fighter that is already far north of the hold line should
            # stop trying to rebuild the NS patrol rectangle first and instead
            # fall back to the dedicated solo defensive leash.
            return None

        # Root-cause fix:
        # Defensive preserve previously used initial north as the lateral anchor.
        # Left/right formations share similar initial north, so broken/late defensive
        # patrol collapsed different formations onto the same hold line, producing
        # southeast drift, pair merging, and circular loiter. Use initial east as the
        # lateral anchor and keep north as the defensive depth axis.
        initial_anchor_east_km = float(np.mean([
            float(getattr(self, '_friendly_initial_east_m', {}).get(aid, my_east_m)) / 1000.0
            for aid in alive_members
        ]))
        if mission_guard_no_enemy:
            cached_anchor = patrol_state.get("no_enemy_anchor_east_km", None)
            if cached_anchor is None or not np.isfinite(float(cached_anchor)):
                cached_anchor = initial_anchor_east_km
            patrol_state["no_enemy_anchor_east_km"] = float(cached_anchor)
            anchor_east_km = float(cached_anchor)
        else:
            patrol_state.pop("no_enemy_anchor_east_km", None)
            cached_anchor = patrol_state.get("anchor_east_km", initial_anchor_east_km)
            if cached_anchor is None or not np.isfinite(float(cached_anchor)):
                cached_anchor = initial_anchor_east_km
            anchor_bias_east_km = initial_anchor_east_km
            if target is not None and getattr(target, 'is_alive', False):
                try:
                    _, target_east_m = self._get_aircraft_ne_position_m(target)
                    target_east_km = float(target_east_m) / 1000.0
                    if np.isfinite(target_east_km) and target_zone in ('HIGH', 'MEDIUM'):
                        target_bias_weight = 0.08 if recreate_posture_hold else 0.18
                        anchor_bias_east_km = float(np.clip(
                            (1.0 - target_bias_weight) * initial_anchor_east_km + target_bias_weight * target_east_km,
                            initial_anchor_east_km - 6.0,
                            initial_anchor_east_km + 6.0,
                        ))
                except Exception:
                    pass
            anchor_smooth_weight = 0.07 if recreate_posture_hold else 0.15
            anchor_east_km = float(np.clip(
                (1.0 - anchor_smooth_weight) * float(cached_anchor) + anchor_smooth_weight * anchor_bias_east_km,
                initial_anchor_east_km - 8.0,
                initial_anchor_east_km + 8.0,
            ))
            patrol_state["anchor_east_km"] = anchor_east_km

        # Defensive patrol must be north-south long and east-west narrow.
        # The previous implementation flipped the axes and kept aircraft
        # oscillating east-west at a nearly fixed hold line.
        longitudinal_leg_halfspan_km = 10.0 if mission_guard_no_enemy else (16.0 if len(alive_members) >= 2 else 12.0)
        lateral_lane_spacing_km = 8.0 if len(alive_members) >= 2 else 0.0
        switch_hold_s = 14.0 if recreate_posture_hold else (10.0 if mission_guard_no_enemy else 6.0)
        last_switch_time = float(patrol_state.get("last_switch_time", -1e9))
        leg_sign = 1 if patrol_state.get("leg_sign", 1) >= 0 else -1
        north_upper_bound_km = hold_depth_center_km + longitudinal_leg_halfspan_km * 0.92
        north_lower_bound_km = hold_depth_center_km - longitudinal_leg_halfspan_km * 0.92
        if my_north_km >= (hold_depth_center_km + longitudinal_leg_halfspan_km * 1.05):
            leg_sign = -1
            patrol_state["last_switch_time"] = current_time_s
        elif my_north_km <= (hold_depth_center_km - longitudinal_leg_halfspan_km * 1.05):
            leg_sign = 1
            patrol_state["last_switch_time"] = current_time_s
        elif (not recreate_posture_hold) and (current_time_s - last_switch_time) >= switch_hold_s:
            if leg_sign > 0 and my_north_km >= north_upper_bound_km:
                leg_sign = -1
                patrol_state["last_switch_time"] = current_time_s
            elif leg_sign < 0 and my_north_km <= north_lower_bound_km:
                leg_sign = 1
                patrol_state["last_switch_time"] = current_time_s
        patrol_state["leg_sign"] = leg_sign

        lane_offset_km = 0.0
        if len(alive_members) >= 2:
            if agent_id == alive_members[0]:
                lane_offset_km = -lateral_lane_spacing_km / 2.0
            elif agent_id == alive_members[1]:
                lane_offset_km = lateral_lane_spacing_km / 2.0
        slot_north_km = float(np.clip(hold_depth_center_km + leg_sign * longitudinal_leg_halfspan_km, 82.0, 116.0))
        if winchester_relief_handoff:
            leg_sign = -1
            patrol_state["leg_sign"] = -1
            slot_north_km = float(np.clip(hold_depth_center_km - longitudinal_leg_halfspan_km, 82.0, 116.0))
        if any(tag in reason_text for tag in ("enemy_regroup_north", "enemy_regroup_no_chase", "low_zone_hold_line")):
            leg_sign = -1
            patrol_state["leg_sign"] = -1
            slot_north_km = float(np.clip(hold_depth_center_km - longitudinal_leg_halfspan_km, 82.0, 108.0))
        slot_east_km = float(anchor_east_km + lane_offset_km)
        slot_error_km = float(np.hypot(my_east_km - slot_east_km, my_north_km - slot_north_km))
        north_error_km = float(my_north_km - slot_north_km)
        lane_error_km = float(my_east_km - slot_east_km)
        terminal_capture_km = float(max(2.5, min(6.0, longitudinal_leg_halfspan_km * 0.30)))
        lateral_soft_lock_km = float(max(1.4, min(3.2, (lateral_lane_spacing_km * 0.28) if lateral_lane_spacing_km > 0.0 else 1.8)))
        lateral_hard_lock_km = float(max(2.8, lateral_soft_lock_km + 1.6))
        along_track_remaining_km = float(abs(slot_north_km - my_north_km))
        deep_axis_recovery = bool(
            abs(north_error_km) >= max(18.0, longitudinal_leg_halfspan_km * 0.90)
        )
        pair_recovery_trigger = bool(
            len(alive_members) >= 2
            and my_north_km >= (hold_depth_center_km + max(24.0, longitudinal_leg_halfspan_km * 1.45))
            and (
                deep_axis_recovery
                or abs(lane_error_km) >= max(8.0, lateral_hard_lock_km + 4.0)
            )
            and any(
                tag in reason_text
                for tag in (
                    "latched_defense",
                    "enemy_regroup",
                    "enemy_retreat",
                    "enemy_regroup_no_chase",
                    "enemy_retreat_guard",
                    "low_zone_forward_commit",
                    "medium_zone_forward_commit",
                    "low_zone_boundary",
                    "medium_zone_boundary",
                    "high_zone_boundary",
                    "overextended",
                )
            )
        )
        pair_recovery_mode = bool(patrol_state.get("pair_recovery_mode", False))
        if pair_recovery_trigger:
            pair_recovery_mode = True
        elif pair_recovery_mode:
            recovery_release_ready = bool(
                my_north_km <= (hold_depth_center_km + max(8.0, longitudinal_leg_halfspan_km * 0.55))
                and abs(lane_error_km) <= max(10.0, lateral_hard_lock_km + 4.0)
            )
            if recovery_release_ready:
                pair_recovery_mode = False
        patrol_state["pair_recovery_mode"] = pair_recovery_mode

        target_east_km = slot_east_km
        target_north_km = slot_north_km
        no_chase_patrol = any(
            tag in reason_text
            for tag in (
                "enemy_regroup_phase_hold",
                "enemy_regroup_north",
                "enemy_regroup_no_chase",
                "winchester_guard",
                "solo_winchester",
                "pair_winchester",
                "pair_low_inventory",
                "latched_defense",
            )
        )
        # When the patrol pair has already overrun far beyond the hold line,
        # forcing a direct capture to the slot is more important than first
        # solving lane error. Otherwise the aircraft can keep flying north
        # for a long time while trying to "re-enter the lane".
        if pair_recovery_mode:
            recovery_lane_limit_km = 5.0 if recreate_posture_hold else 7.5
            recovery_lane_center_km = float(anchor_east_km + lane_offset_km)
            target_east_km = float(np.clip(
                my_east_km,
                recovery_lane_center_km - recovery_lane_limit_km,
                recovery_lane_center_km + recovery_lane_limit_km,
            ))
            target_north_km = float(np.clip(
                hold_depth_center_km + max(2.0, longitudinal_leg_halfspan_km * 0.15),
                84.0,
                112.0,
            ))
            target_heading = self._heading_to_point(
                env,
                agent_id,
                target_north_km * 1000.0,
                target_east_km * 1000.0,
            )
        elif deep_axis_recovery:
            target_heading = self._heading_to_point(
                env,
                agent_id,
                slot_north_km * 1000.0,
                slot_east_km * 1000.0,
            )
        elif no_chase_patrol:
            base_heading = 0.0 if leg_sign > 0 else 180.0
            corridor_gain = 1.8 if recreate_posture_hold else 2.6
            corridor_limit = 8.0 if recreate_posture_hold else 12.0
            corridor_correction_deg = float(np.clip(lane_error_km * corridor_gain, -corridor_limit, corridor_limit))
            if leg_sign > 0:
                target_heading = float((base_heading - corridor_correction_deg) % 360.0)
            else:
                target_heading = float((base_heading + corridor_correction_deg) % 360.0)
        elif abs(lane_error_km) >= lateral_hard_lock_km:
            lookahead_north_km = float(np.clip(
                my_north_km + leg_sign * max(2.5, min(6.0, along_track_remaining_km * 0.35)),
                82.0,
                116.0,
            ))
            target_heading = self._heading_to_point(
                env,
                agent_id,
                lookahead_north_km * 1000.0,
                slot_east_km * 1000.0,
            )
        elif along_track_remaining_km <= terminal_capture_km:
            target_heading = self._heading_to_point(
                env,
                agent_id,
                slot_north_km * 1000.0,
                slot_east_km * 1000.0,
            )
        elif abs(lane_error_km) > lateral_soft_lock_km:
            base_heading = 0.0 if leg_sign > 0 else 180.0
            correction_gain = 2.4 if recreate_posture_hold else 4.0
            correction_limit = 10.0 if recreate_posture_hold else 18.0
            correction_deg = float(np.clip(lane_error_km * correction_gain, -correction_limit, correction_limit))
            if leg_sign > 0:
                target_heading = float((base_heading - correction_deg) % 360.0)
            else:
                target_heading = float((base_heading + correction_deg) % 360.0)
        else:
            target_heading = 0.0 if leg_sign > 0 else 180.0
        executor = getattr(self, 'executor', None)
        if executor is not None and hasattr(executor, '_smooth_heading'):
            smooth_alpha = 0.52 if pair_recovery_mode else (0.08 if no_chase_patrol else (0.12 if recreate_posture_hold else 0.22))
            target_heading = float(
                executor._smooth_heading(
                    f"def_guard:{formation_label}:{agent_id}",
                    target_heading,
                    alpha=smooth_alpha,
                )
            )
        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
        alt_cmd = 8 if current_altitude < 7500.0 else 7
        if mission_guard_no_enemy:
            if slot_error_km > 12.0 or abs(north_error_km) > 8.0:
                vel_cmd = 5
            elif slot_error_km > 5.0:
                vel_cmd = 4
            else:
                vel_cmd = 4
        elif pair_recovery_mode:
            vel_cmd = 6
        elif deep_axis_recovery or abs(lane_error_km) > lateral_hard_lock_km or slot_error_km > 20.0 or abs(north_error_km) > 12.0:
            vel_cmd = 6
        elif slot_error_km > 8.0:
            vel_cmd = 5
        else:
            vel_cmd = 4 if len(alive_members) >= 2 else 3
        if recreate_posture_hold and not deep_axis_recovery:
            vel_cmd = min(vel_cmd, 4 if slot_error_km <= 12.0 else 5)
        if recreate_recovering and slot_error_km <= 10.0 and abs(lane_error_km) <= lateral_hard_lock_km:
            vel_cmd = min(vel_cmd, 4)
        hdg_cmd = self._get_heading_cmd(env, agent_id, target_heading)

        if env.current_step % 80 == 0:
            logging.info(
                "🛡️ [DEFENSIVE_PATROL] %s pair=%s zone=%s pattern=NS_RECT anchor_east=%.1fkm hold_north=%.1fkm "
                "span=(EW%.1f,NS%.1f)km slot=(E%.1f,N%.1f)km current=(E%.1f,N%.1f)km leg=%+d lane=%.1fkm recover=%s pair_recover=%s recreate_hold=%s dist=%s reason=%s",
                agent_id,
                formation_label,
                target_zone,
                anchor_east_km,
                hold_depth_center_km,
                max(lateral_lane_spacing_km, 2.0),
                longitudinal_leg_halfspan_km * 2.0,
                slot_east_km,
                slot_north_km,
                my_east_km,
                my_north_km,
                int(leg_sign),
                lane_error_km,
                "Y" if deep_axis_recovery else "N",
                "Y" if pair_recovery_mode else "N",
                "Y" if recreate_posture_hold else "N",
                f"{float(distance) / 1000.0:.1f}km" if distance is not None and np.isfinite(distance) else "-",
                reason or "defensive_patrol",
            )
            self._log_key_event(
                env,
                agent_id,
                "守区巡逻",
                current_time=current_time_s,
                当前阶段=(self.state_manager.get_agent_phase(agent_id).value if hasattr(self, 'state_manager') else None),
                当前战术=getattr(self, 'selected_tactic', None),
                进入函数="_build_red_defensive_patrol_command",
                执行机动="DEFENSIVE_PATROL",
                当前状态=f"pair={formation_label}, zone={target_zone}, leg={int(leg_sign)}, lane={lane_error_km:.1f}km, recover={deep_axis_recovery}, pair_recover={pair_recovery_mode}, recreate_hold={recreate_posture_hold}, no_chase={no_chase_patrol}",
                目标点=f"slot(E={slot_east_km:.1f},N={slot_north_km:.1f})km hdg={target_heading:.1f}",
                当前指令=str((alt_cmd, hdg_cmd, vel_cmd)),
                退出条件=f"reason_change / enemy_alive_change / slot_capture, reason={reason or 'defensive_patrol'}",
                是否满足退出="否",
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _build_solo_defensive_hold_command(self, env, agent_id: str, target=None, distance: float = None, reason: str = ""):
        aircraft = self._get_aircraft_from_any_collection(env, agent_id)
        if aircraft is None or not getattr(aircraft, 'is_alive', False):
            return None
        if not str(agent_id).startswith('A'):
            return None
        if self._get_formation_alive_count(env, agent_id) > 1:
            return None
        if (
            hasattr(self, '_build_cap_solo_defensive_hold_command')
            and hasattr(self, '_resolve_maneuver_intent_command')
        ):
            current_time = float(getattr(env, 'current_step', 0)) * float(getattr(env, 'time_interval', 0.2))
            target_info = self._resolve_preserve_target(env, target=target) if hasattr(self, '_resolve_preserve_target') else {"track_id": "", "x_km": float("nan"), "y_km": float("nan"), "source": "none"}
            distance_km = self._normalize_preserve_distance_km(env, agent_id, target_info, distance=distance) if hasattr(self, '_normalize_preserve_distance_km') else (float(distance) / 1000.0 if distance is not None and np.isfinite(distance) and float(distance) > 1000.0 else float(distance) if distance is not None and np.isfinite(distance) else None)
            intent = self._build_cap_solo_defensive_hold_command(
                env,
                agent_id,
                target_info,
                distance_km=distance_km,
                reason=reason,
                current_time=current_time,
            )
            if intent is not None:
                return self._resolve_maneuver_intent_command(env, agent_id, [intent], current_time=current_time)

        if not hasattr(self, '_solo_defensive_hold_state'):
            self._solo_defensive_hold_state = {}
        patrol_state = self._solo_defensive_hold_state.setdefault(agent_id, {})

        # Solo defensive hold must also scrub CAP engagement state immediately.
        self._mark_cap_defensive_engagement_hold(
            env,
            [agent_id],
            reason=reason or "solo_defensive_hold",
            hold_steps=30,
        )

        my_pos = aircraft.get_position()
        current_north = float(my_pos[0])
        current_east = float(my_pos[1])
        anchor_east = float(getattr(self, '_friendly_initial_east_m', {}).get(agent_id, current_east))
        anchor_north = float(getattr(self, '_friendly_initial_north_m', {}).get(agent_id, current_north))
        reason_text = str(reason or "")
        winchester_relief_handoff = "winchester_relief_handoff" in reason_text
        floor_overrun = "bvr_floor_overrun" in reason_text
        aggressive_breakaway = any(
            tag in reason_text
            for tag in ("close_merge", "forward_commit", "enemy_retreat", "enemy_regroup", "guidance_commit_overrun", "overextended", "bvr_floor_overrun")
        )
        hold_east_limit_m = 12000.0 if aggressive_breakaway else 18000.0
        hold_east = float(np.clip(anchor_east, current_east - hold_east_limit_m, current_east + hold_east_limit_m))
        hold_offset_m = 12000.0 if aggressive_breakaway else (18000.0 if winchester_relief_handoff else 24000.0)
        hard_floor_m = float(self._get_friendly_bvr_hard_floor_m(agent_id))
        hold_floor_m = 28000.0 if aggressive_breakaway else (30000.0 if winchester_relief_handoff else 35000.0)
        hold_floor_m = max(hold_floor_m, hard_floor_m)
        hold_north = float(np.clip(max(anchor_north + hold_offset_m, hold_floor_m), 25000.0, 85000.0))
        lateral_error = abs(current_east - hold_east)
        north_overrun_m = current_north - hold_north
        entry_floor_m = float(self._get_friendly_bvr_entry_floor_m(agent_id))
        release_floor_m = entry_floor_m + 5000.0
        floor_distance_m = float(distance) if distance is not None and np.isfinite(distance) else None
        floor_state_key = f"floor_escape_active_{agent_id}"
        floor_escape_active = bool(patrol_state.get(floor_state_key, False))
        if floor_overrun and floor_distance_m is not None:
            if floor_escape_active:
                floor_escape_active = floor_distance_m < release_floor_m
            else:
                floor_escape_active = floor_distance_m <= entry_floor_m
        else:
            floor_escape_active = False
        patrol_state[floor_state_key] = floor_escape_active
        retreat_needed = bool(north_overrun_m > 5000.0 or floor_escape_active)
        hard_retreat_needed = bool(
            north_overrun_m > 18000.0
            or (floor_escape_active and floor_distance_m is not None and floor_distance_m <= hard_floor_m + 2000.0)
        )

        # Solo guard should not keep drawing large circles on the defensive boundary.
        # Use a hysteresis-based southbound recovery corridor first, then switch to a
        # narrow north/south shuttle around the hold line.
        recovery_state_key = f"recovery_mode_{agent_id}"
        leg_state_key = f"leg_dir_{agent_id}"
        recovery_active = bool(patrol_state.get(recovery_state_key, False))
        recovery_trigger = bool(
            hard_retreat_needed
            or north_overrun_m > 9000.0
            or lateral_error > 10000.0
            or floor_escape_active
        )
        recovery_release = bool(
            north_overrun_m <= 2500.0
            and lateral_error <= 4500.0
            and not floor_escape_active
        )
        if recovery_active:
            recovery_active = not recovery_release
        elif recovery_trigger:
            recovery_active = True
        patrol_state[recovery_state_key] = recovery_active

        lane_bias_east = float(np.clip((hold_east - current_east) / 420.0, -26.0, 26.0))
        if recovery_active:
            if floor_escape_active:
                recovery_base = 180.0 - float(np.clip((hold_east - current_east) / 1500.0, -3.0, 3.0))
            elif floor_overrun:
                recovery_base = 180.0 - float(np.clip((hold_east - current_east) / 1200.0, -5.0, 5.0))
            else:
                recovery_base = 180.0 - lane_bias_east
            if hard_retreat_needed:
                target_heading = float(recovery_base % 360.0)
            else:
                if floor_escape_active:
                    target_heading = float(recovery_base % 360.0)
                elif floor_overrun:
                    target_heading = float(recovery_base % 360.0)
                else:
                    corridor_north = max(25000.0, min(hold_north - 4000.0, current_north - 12000.0))
                    corridor_heading = float(self._heading_to_point(env, agent_id, corridor_north, hold_east))
                    corridor_delta = float(np.clip(self._normalize_angle_diff(corridor_heading - 180.0), -28.0, 28.0))
                    target_heading = float((180.0 + corridor_delta) % 360.0)
            leg_dir = "south"
        else:
            patrol_center_north = hold_north - (3000.0 if winchester_relief_handoff else 1500.0)
            upper_turn_north = patrol_center_north + 3000.0
            lower_turn_north = patrol_center_north - 7000.0
            leg_dir = str(patrol_state.get(leg_state_key, "") or "")
            if leg_dir not in ("north", "south"):
                leg_dir = "south" if current_north > patrol_center_north else "north"
            if current_north >= upper_turn_north:
                leg_dir = "south"
            elif current_north <= lower_turn_north:
                leg_dir = "north"
            patrol_state[leg_state_key] = leg_dir
            shuttle_bias_east = float(np.clip((hold_east - current_east) / 600.0, -18.0, 18.0))
            base_heading = 0.0 if leg_dir == "north" else 180.0
            target_heading = float((base_heading - shuttle_bias_east) % 360.0)

        current_altitude = float(aircraft.get_property_value(c.position_h_sl_m))
        alt_cmd = 8 if current_altitude < 7500.0 else 7
        if floor_escape_active:
            vel_cmd = 6
        elif hard_retreat_needed or "bvr_floor_overrun" in reason_text:
            vel_cmd = 6
        elif recovery_active and north_overrun_m > 14000.0:
            vel_cmd = 5
        elif recovery_active or lateral_error > 12000.0:
            vel_cmd = 5 if aggressive_breakaway else 4
        else:
            vel_cmd = 3
        slot_error_m = float(np.hypot(current_east - hold_east, current_north - hold_north))
        heading_state_key = f"heading_target_{agent_id}"
        previous_target_heading = patrol_state.get(heading_state_key, None)
        if (
            previous_target_heading is not None
            and np.isfinite(float(previous_target_heading))
            and not hard_retreat_needed
            and not floor_escape_active
            and slot_error_m <= 22000.0
        ):
            heading_delta = float(self._normalize_angle_diff(target_heading - float(previous_target_heading)))
            if abs(heading_delta) <= 36.0:
                target_heading = float((float(previous_target_heading) + 0.45 * heading_delta) % 360.0)
        patrol_state[heading_state_key] = float(target_heading)
        hdg_cmd = self._get_heading_cmd(env, agent_id, target_heading)

        if env.current_step % 80 == 0:
            target_zone = self._classify_target_risk_zone(target, env=env) if target is not None else "UNKNOWN"
            dist_text = f"{float(distance) / 1000.0:.1f}km" if distance is not None and np.isfinite(distance) else "-"
            logging.info(
                f"🛡️ [DEFENSIVE_LEASH] {agent_id} zone={target_zone} dist={dist_text} "
                f"hold=(E{hold_east/1000.0:.1f},N{hold_north/1000.0:.1f})km current=(E{current_east/1000.0:.1f},N{current_north/1000.0:.1f})km "
                f"hard_retreat={'Y' if hard_retreat_needed else 'N'} recovery={'Y' if recovery_active else 'N'} floor_escape={'Y' if floor_escape_active else 'N'} "
                f"leg={leg_dir} reason={reason or 'solo_hold'}"
            )
        return alt_cmd, hdg_cmd, vel_cmd

    def _set_agent_tactic(self, agent_id: str, tactic):
        formation_label = self._get_formation_label(agent_id, self._get_reference_env())
        previous_tactic = self._formation_tactic_overrides.get(formation_label, self.selected_tactic)
        self._formation_tactic_overrides[formation_label] = tactic

        if previous_tactic != tactic:
            executor = getattr(self, 'executor', None)
            if executor and hasattr(executor, 'on_formation_tactic_changed'):
                try:
                    executor.on_formation_tactic_changed(formation_label, previous_tactic, tactic)
                except Exception:
                    pass

    def _get_agent_tactic(self, agent_id: str):
        formation_label = self._get_formation_label(agent_id, self._get_reference_env())
        if formation_label in self._formation_tactic_overrides:
            return self._formation_tactic_overrides[formation_label]
        return self.selected_tactic

    def _iter_agent_collections_for_enemy_checks(self, env):
        yield from self._iter_agent_collections(env)

    def _has_alive_enemy_anywhere(self, env, agent_id: str) -> bool:
        global_enemy_alive = self._get_global_alive_enemy_count(env, agent_id)
        if global_enemy_alive >= 0:
            return bool(global_enemy_alive > 0)
        enemy_prefix = 'B' if str(agent_id).startswith('A') else 'A'
        seen_agent_ids = set()

        for collection in self._iter_agent_collections_for_enemy_checks(env):
            for other_id, other_aircraft in collection.items():
                other_id = str(other_id)
                if not other_id.startswith(enemy_prefix) or other_id in seen_agent_ids:
                    continue
                seen_agent_ids.add(other_id)
                if getattr(other_aircraft, 'is_alive', False):
                    return True
        return False

    def _transition_formation_to_rtb(self, env, agent_id: str, reason: str = ""):
        formation_label = self._get_formation_label(agent_id, env)
        formation_agents = [
            aid for aid in self._get_formation_agents(agent_id, env)
            if self._get_aircraft_from_any_collection(env, aid) is not None
        ]
        alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)

        if str(agent_id).startswith('A') and alive_enemy_exists:
            defensive_state = self._evaluate_red_defensive_posture(
                env,
                agent_id,
                current_phase_name="transition_rtb",
                reason=reason or "transition_rtb",
            )
            if defensive_state.get("preserve", False):
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason=defensive_state.get("reason", reason or "transition_rtb"),
                )
                logging.info(
                    "🛡️ [%s] 保留守区，不转RTB攻击链 reason=%s posture=%s",
                    formation_label,
                    reason or "transition_rtb",
                    defensive_state.get("reason", "guard"),
                )
                return

            override_plan = self._build_red_reengage_override_plan(
                env,
                agent_id,
                current_phase_name="transition_rtb",
                reason=reason or "transition_rtb",
                allow_template_reset=True,
            )
            formation_agents = override_plan.get("formation_agents", formation_agents) or formation_agents
            alive_formation_agents = [
                aid for aid in formation_agents
                if (
                    (lambda aircraft: aircraft is not None and getattr(aircraft, 'is_alive', False))(
                        self._get_aircraft_from_any_collection(env, aid)
                    )
                )
            ]
            replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
            reason_text = f"，原因: {reason}" if reason else ""
            if replacement_tactic == 'DEFENSIVE_GUARD':
                self._deescalate_red_formation_to_defense(
                    env,
                    agent_id,
                    reason=str(override_plan.get("forward_commit_reason", "") or "transition_rtb_guard"),
                )
                logging.warning(f"[RTB拦截] [{formation_label}] 改为DEFENSIVE_GUARD{reason_text}")
                return

            for aid in formation_agents:
                self.returning_agents.discard(aid)
                self.set_agent_second_attack(aid, True)
                if hasattr(self, 'missile_launched'):
                    self.missile_launched[aid] = False
                state_manager = getattr(self, 'state_manager', None)
                if state_manager and hasattr(state_manager, 'missile_launched'):
                    state_manager.missile_launched[aid] = False
                if hasattr(self, '_selected_second_attack_tactic'):
                    self._selected_second_attack_tactic.pop(aid, None)
                executor = getattr(self, 'executor', None)
                if executor is not None and hasattr(executor, 'tactical_turn_states'):
                    executor.tactical_turn_states.pop(aid, None)
            self._second_attack_tactic_selected_by_formation.pop(formation_label, None)
            for aid in alive_formation_agents:
                self._set_agent_tactic(aid, replacement_tactic)

            if replacement_tactic == 'FORMATION_RESET' and alive_formation_agents:
                formation_manager = self._get_formation_reset_manager(agent_id)
                if hasattr(formation_manager, 'start_formation_reset') and (not formation_manager.is_reset_active()):
                    try:
                        formation_manager.start_formation_reset(env, env.current_step * env.time_interval, alive_formation_agents)
                    except Exception:
                        pass

            logging.warning(f"[RTB拦截] [{formation_label}] 改为{replacement_tactic}{reason_text}")
            return

        self._set_agent_tactic(agent_id, 'TACTICAL_TURN')

        for aid in formation_agents:
            self.returning_agents.add(aid)
            self.set_agent_second_attack(aid, False)

            if hasattr(self, 'missile_launched'):
                self.missile_launched[aid] = False

            state_manager = getattr(self, 'state_manager', None)
            if state_manager and hasattr(state_manager, 'missile_launched'):
                state_manager.missile_launched[aid] = False

            if hasattr(self, '_selected_second_attack_tactic'):
                self._selected_second_attack_tactic.pop(aid, None)

        self._second_attack_tactic_selected_by_formation.pop(formation_label, None)

        reason_text = f"，原因: {reason}" if reason else ""
        logging.info(f"[返航切换] [{formation_label}] 切换到TACTICAL_TURN{reason_text}")
    
    def _update_phase(self, env, agent_id: str, current_time: float):
        if hasattr(_ttrfh, '_update_phase_v2'):
            return _ttrfh._update_phase_v2(self, env, agent_id, current_time)
        return _ttrfh._update_phase(self, env, agent_id, current_time)

    def _select_tactic_at_phase(self, env, agent_id: str):
        return _ttrfh._select_tactic_at_phase(self, env, agent_id)

    def _decide_at_lr(self, env, agent_id: str):
        return _ttrfh._decide_at_lr(self, env, agent_id)

    def _node_decision_at_node(self, env, agent_id: str, node: str, current_time: float):
        return _ttrfh._node_decision_at_node(self, env, agent_id, node, current_time)

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
    
    def _get_dynamic_velocity_cmd(self, env, agent_id: str) -> int:
        return _ttrfh._get_dynamic_velocity_cmd(self, env, agent_id)

    def _calculate_distance_between(self, aircraft1, aircraft2) -> float:
        """计算距离 - 委托给工具类"""
        return TacticalUtils.calculate_distance_between(aircraft1, aircraft2)

    def _get_current_heading_deg(self, env, agent_id: str) -> float:
        """统一当前航向来源，优先使用真航向避免大坡度时 yaw 与地理航向失配。"""
        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return 0.0
        return float(TacticalUtils.get_aircraft_heading_deg(aircraft))
    
    def _stabilize_antipodal_turn(self, agent_id: str, heading_diff_deg: float) -> float:
        """Keep turn direction stable when desired heading is near exactly opposite (+/-180 deg)."""
        if not hasattr(self, '_heading_turn_memory'):
            self._heading_turn_memory = {}

        heading_diff_deg = float(heading_diff_deg)
        abs_diff = abs(heading_diff_deg)
        turn_sign = self._heading_turn_memory.get(agent_id)

        if abs(abs_diff - 180.0) <= 12.0:
            if turn_sign not in (-1, 1):
                turn_sign = 1 if heading_diff_deg >= 0.0 else -1
                self._heading_turn_memory[agent_id] = turn_sign
            return abs_diff * float(turn_sign)

        if abs_diff < 150.0:
            self._heading_turn_memory[agent_id] = 1 if heading_diff_deg >= 0.0 else -1

        return heading_diff_deg

    def _convert_heading_to_index(self, heading_diff: float, agent_id: str = None) -> int:
        """航向转指令 - 对 A 机在 ±180° 附近做防抖后再统一离散化。"""
        heading_diff_deg = float(np.rad2deg(float(heading_diff)))
        if agent_id:
            heading_diff_deg = self._stabilize_antipodal_turn(str(agent_id), heading_diff_deg)
        return TacticalUtils.convert_heading_to_index(np.deg2rad(heading_diff_deg))
    
    def _convert_altitude_to_index(self, altitude_cmd_value: float) -> int:
        """高度转指令 - 委托给工具类"""
        return TacticalUtils.convert_altitude_to_index(altitude_cmd_value)
    
    def _format_phase_name(self, agent_id: str, phase) -> str:
        return _ttrfh._format_phase_name(self, agent_id, phase)

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

    def _heading_to_point(self, env, agent_id: str, target_north: float, target_east: float) -> float:
        """Convert a north/east target point into a navigation heading."""
        aircraft = env.agents.get(agent_id)
        if aircraft is None:
            return 0.0
        current_north, current_east = self._get_aircraft_ne_position_m(aircraft)
        delta_north = float(target_north) - current_north
        delta_east = float(target_east) - current_east
        return float((np.degrees(np.arctan2(delta_east, delta_north)) + 360.0) % 360.0)

    def _get_enemy_bearing(self, env, agent_id: str) -> float:
        return TacticalUtils.get_enemy_bearing(env, agent_id)

    def _get_wingman_phase_by_distance(self, distance: float) -> TacticalPhase:
        """Map distance to the wingman phase thresholds used by TacticalTask."""
        distance = float(distance)
        adjusted_distance = distance
        if distance >= self.tactical_distances['NLT']:
            return TacticalPhase.BEYOND_NLT
        if distance >= self.tactical_distances['MELD']:
            return TacticalPhase.NLT_MELD
        if distance >= self.tactical_distances['MTR']:
            return TacticalPhase.MELD_MTR
        if adjusted_distance >= self.tactical_distances['LR']:
            return TacticalPhase.MTR_LR
        if adjusted_distance >= self.tactical_distances['TR']:
            return TacticalPhase.LR_TR
        if adjusted_distance >= self.tactical_distances['DOR']:
            return TacticalPhase.TR_DOR
        if adjusted_distance >= self.tactical_distances['DR']:
            return TacticalPhase.DOR_DR
        if adjusted_distance >= self.tactical_distances['MAR']:
            return TacticalPhase.DR_MAR
        return TacticalPhase.BEYOND_MAR

    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float, duration=10.0) -> tuple:
        return self.maneuver_lib.maintain_heading_precise(env, agent_id, target_heading, duration)

    def _execute_short_skate_precise(self, env, agent_id: str, current_time: float, skate_direction: str = None) -> tuple:
        direction = 'auto' if skate_direction is None else skate_direction
        return self.maneuver_lib.execute_short_skate_precise(env, agent_id, current_time, direction)

    def _establish_rear_formation(self, env, agent_id: str, current_time: float) -> tuple:
        return self.maneuver_lib.establish_rear_formation(env, agent_id, current_time)

    def _get_heading_cmd(self, env, agent_id: str, target_heading: float) -> int:
        current_heading = self._get_current_heading_deg(env, agent_id)
        heading_diff = self._normalize_angle_diff(float(target_heading) - current_heading)
        heading_diff = self._stabilize_antipodal_turn(agent_id, heading_diff)
        if abs(heading_diff) <= 5.0:
            return 8
        return int(self._convert_heading_to_index(np.deg2rad(heading_diff), agent_id=agent_id))

    def _execute_tactical_crank(self, env, agent_id: str, direction: str = 'left', climb: bool = False, angle: float = 30.0) -> tuple:
        angle = abs(float(angle))
        heading_offset = -angle if direction == 'left' else angle
        altitude_cmd_id = 11 if climb else 7
        heading_cmd_id = int(self._convert_heading_to_index(np.deg2rad(heading_offset)))
        return altitude_cmd_id, heading_cmd_id, 3

    def _should_block_rtb_lock_for_second_attack(self, env, agent_id: str, current_time: float, reason: str = "") -> bool:
        return bool(agent_id.startswith('A') and self._has_alive_enemy_anywhere(env, agent_id))

    def normalize_action(self, env, agent_id, action):
        return _ttrfh.normalize_action(self, env, agent_id, action)

    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int) -> np.ndarray:
        return _ttrfh._use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

    def _handle_missile_launches(self, env, current_time: float):
        return _ttrfh._handle_missile_launches(self, env, current_time)

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        return _ttrfh._launch_missile(self, env, agent_id, target, current_time)

    def _normalize_angle_diff(self, angle_diff):
        """标准化角度差到[-180, 180]区间"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff

    # ==================== 新增：二次进攻智能航向调整系统 ====================
    
    def _start_formation_reset_procedure(self, env, agent_id: str, current_time: float):
        return _ttrh._start_formation_reset_procedure(self, env, agent_id, current_time)

    def _handle_dr_reengage_unified(self, env, agent_id: str, current_time: float, 
                                   enemies_alive: int, threat_level: float):
        return _ttrh._handle_dr_reengage_unified(self, env, agent_id, current_time, enemies_alive, threat_level)
    
    def _execute_formation_reset_procedure(self, env, agent_id: str) -> tuple:
        return _ttrh._execute_formation_reset_procedure(self, env, agent_id)
    
    def _execute_unified_second_attack(self, env, agent_id: str) -> tuple:
        return _ttrh._execute_unified_second_attack(self, env, agent_id)
        
    def _execute_intelligent_reengagement(self, env, agent_id: str, current_time: float) -> tuple:
        """执行智能重新交战回转 - 动态航向调整"""
        if not hasattr(self, 'intelligent_reengagement_states'):
            return (7, 8, 3)  # 默认直飞
            
        state = self.intelligent_reengagement_states.get(agent_id)
        if not state:
            return (7, 8, 3)
            
        current_heading = self._get_current_heading_deg(env, agent_id)
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        target_heading = state['target_heading']
        
        # 计算航向差
        heading_diff = self._normalize_angle_diff(target_heading - current_heading)
        
        # 高度保护优先
        # 🔥 使用动态速度管理，根据当前速度选择合适的速度指令
        vel_cmd = self._get_dynamic_velocity_cmd(env, agent_id)
        
        if current_altitude < 4000:
            alt_cmd = 8  # 轻微爬升
            # 爬升时额外加速补偿（在动态速度基础上+1，但不超过6）
            vel_cmd = min(vel_cmd + 1, 6)
        elif current_altitude < 3000:
            alt_cmd = 9  # 爬升
            # 爬升时额外加速补偿（在动态速度基础上+1，但不超过6）
            vel_cmd = min(vel_cmd + 1, 6)
        else:
            alt_cmd = 7  # 保持
            # 使用动态速度（已在上面计算）
            
        # 动态航向调整
        if abs(heading_diff) <= state['completion_threshold']:
            # 完成调整
            if not state['second_attack_ready']:
                state['second_attack_ready'] = True
                logging.info(f"[二次进攻回转] {agent_id} 智能回转完成，最终航向: {current_heading:.1f}° (目标0°)")
                
                # 选择二次进攻战术
                second_attack_tactic = self._select_second_attack_tactic()
                self.selected_tactic = second_attack_tactic
                self.set_agent_second_attack(agent_id, True)
                logging.info(f"[二次进攻] {agent_id} 开始战术: {second_attack_tactic}")
                
                # 清理状态
                self.intelligent_reengagement_states.pop(agent_id, None)
            # 🔥 高度上升时速度补偿：爬升需要额外能量
            if alt_cmd in [8, 9, 10, 11, 12]:  # 爬升指令
                vel_cmd = min(vel_cmd + 1, 6)  # 爬升时额外加速补偿
            return (alt_cmd, 8, vel_cmd)  # 直飞（带动态速度补偿）
            
        # 进行航向调整
        if int(current_time) % 5 == 0:  # 每5秒输出一次调试信息
            logging.info(f"[二次进攻回转] [{agent_id}] 当前{current_heading:.1f}° -> 目标{target_heading}° (差值{heading_diff:.1f}°)")
            
        if abs(heading_diff) <= 10:
            # 接近目标，精调阶段
            logging.info(f"[二次进攻回转] {agent_id} 航向接近目标({heading_diff:.1f}°)，进入精调阶段")
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
        return _ttrh._check_intelligent_reengagement_active(self, agent_id)
        
    def _select_second_attack_tactic(self, env=None, agent_id=None, enemies_alive=0, threat_level=0.0) -> str:
        return _ttrh._select_second_attack_tactic(self, env, agent_id, enemies_alive, threat_level)

    def _is_second_attack_ready_to_fire(self, agent_id: str = None) -> bool:
        return _ttrh._is_second_attack_ready_to_fire(self, agent_id)

    # ==================== DR决策统一协调系统 ====================
    
    def _handle_dr_decision_unified(self, env, agent_id: str, current_time: float):
        return _ttrh._handle_dr_decision_unified(self, env, agent_id, current_time)
    
    def _handle_dr_retreat_unified(self, env, agent_id: str, current_time: float):
        return _ttrh._handle_dr_retreat_unified(self, env, agent_id, current_time)
    
    def _count_alive_enemies(self, env) -> int:
        return _ttrh._count_alive_enemies(self, env)
    
    def _calculate_comprehensive_threat_level(self, env) -> float:
        return _ttrh._calculate_comprehensive_threat_level(self, env)
    
    def _is_in_rtb_mode(self, env, agent_id: str, current_distance: float, current_time: float) -> bool:
        """
        检测飞机是否处于返航模式 (RTB - Return To Base)
        
        返航检测条件：
        1. 距离超过NLT(120km)且持续增大
        2. 航向背离敌机（朝向后方飞行）
        3. 已经过了某个战术阶段（不是初始状态）
        
        Args:
            env: 环境
            agent_id: 飞机ID
            current_distance: 当前与目标距离
            current_time: 当前时间
            
        Returns:
            True表示处于返航模式
        """
        # 只对我方飞机检测返航
        if not agent_id.startswith('A'):
            return False

        alive_enemy_exists = self._has_alive_enemy_anywhere(env, agent_id)
        current_time = float(getattr(env, 'current_step', 0)) * float(getattr(env, 'time_interval', 0.2))
        guard_release_until = float(getattr(self, '_red_guard_release_until', {}).get(agent_id, -999.0))
        guard_release_active = current_time < guard_release_until
        if alive_enemy_exists:
            if agent_id in self.returning_agents:
                current_tactic_name = str(self._get_agent_tactic(agent_id) or '')
                preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
                try:
                    preserve_defense = preserve_defense or bool(
                        self._evaluate_red_defensive_posture(
                            env,
                            agent_id,
                            current_phase_name="returning_query",
                            reason="returning_query",
                        ).get("preserve", False)
                    )
                except Exception:
                    preserve_defense = bool(guard_release_active or current_tactic_name == 'DEFENSIVE_GUARD')
                if preserve_defense:
                    self._deescalate_red_formation_to_defense(
                        env,
                        agent_id,
                        reason="returning_query",
                        clear_returning=False,
                    )
                    if env.current_step % 60 == 0:
                        logging.info(f"[守区止追] [{agent_id}] 敌机仍存活，保持守区止追，不恢复循环进攻")
                    return False
                override_plan = self._build_red_reengage_override_plan(
                    env,
                    agent_id,
                    current_phase_name="returning_query",
                    reason="returning_query",
                    allow_template_reset=True,
                )
                replacement_tactic = str(override_plan.get("replacement_tactic", 'ADAPTIVE_ATTACK'))
                if replacement_tactic == 'DEFENSIVE_GUARD':
                    self._deescalate_red_formation_to_defense(
                        env,
                        agent_id,
                        reason="returning_query_guard",
                        clear_returning=False,
                    )
                    if env.current_step % 60 == 0:
                        logging.info(f"[解除RTB] [{agent_id}] 敌机仍存活，但当前保持守区防御")
                    return False
                formation_agents = override_plan.get("formation_agents", []) or [
                    aid for aid in self._get_formation_agents(agent_id, env)
                    if aid in getattr(env, 'agents', {})
                ]
                for aid in formation_agents:
                    self.returning_agents.discard(aid)
                    self.set_agent_second_attack(aid, True)
                for aid in formation_agents:
                    self._set_agent_tactic(aid, replacement_tactic)
                if env.current_step % 60 == 0:
                    logging.info(f"[解除RTB] [{agent_id}] 敌机仍存活，清除RTB状态并恢复循环进攻")
            return False

        return agent_id in self.returning_agents
