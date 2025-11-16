"""
通用战术任务类 - 基于原项目DragShootTacticalTask架构
集成战术决策系统，完整控制2v2空战仿真
Version: 2024-10-24-22:48 - 完整战术执行系统
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

from core.threat_evaluator import ThreatEvaluator
from core.situation_evaluator import SituationEvaluator
from core.intent_predictor import IntentPredictor
from core.decision_table import DecisionTable
from core.tactic_selector_v2 import TacticSelectorV2
from core.complete_tactical_system import CompleteTacticalSystem  # 新增：完整智能战术系统
from core.target_assignment import get_fixed_target, get_target_with_fallback  # 新增：固定目标分配
from enemy_ai_adapter import EnemyAIAdapter
from utils.data_logger import DataLogger
from utils.constants import CONTROL_RANGES

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
        self.max_steps = getattr(config, 'max_steps', 1500)  # 300秒
    
    def get_termination(self, task, env, agent_id, info={}):
        """终止条件检查"""
        # 高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} 高度过低: {current_alt:.1f}m")
            return True, False, info
        
        # 时间限制
        if env.current_step >= self.max_steps:
            self.log(f"达到时间限制: {env.current_step} 步")
            return True, False, info
        
        # 极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} 检测到极端状态")
            return True, False, info
        
        # 过载
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} 过载")
            return True, False, info
        
        # 检查双方存活 - 一方全灭才终止
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
    通用战术任务类
    基于MultipleCombatTask，集成战术决策系统
    """
    
    def __init__(self, config, decision_manager=None):
        super().__init__(config)
        
        # 使用自定义终止条件
        self.termination_conditions = [TacticalTermination(self.config)]
        
        # 兼容参数但不再使用外部decision_manager（冗余路径已移除）
        
        # 集成战术系统（新增）
        from core.integrated_tactical_system import IntegratedTacticalSystem
        from core.intent_recognizer import FriendlyIntent

        # 设置我方意图（从配置读取或默认为保守肃清）
        my_intent_str = getattr(config, 'friendly_intent', 'CONSERVATIVE_CLEAR')
        if my_intent_str == 'AGGRESSIVE_CLEAR':
            my_intent = FriendlyIntent.AGGRESSIVE_CLEAR
        elif my_intent_str == 'DEFENSIVE':
            my_intent = FriendlyIntent.DEFENSIVE
        else:
            my_intent = FriendlyIntent.CONSERVATIVE_CLEAR

        self.integrated_system = IntegratedTacticalSystem(my_intent=my_intent)
        logging.info(f"✅ 集成战术系统已启动 (我方意图: {my_intent.value})")

        # 完整智能战术选择系统（新增 - 实现项目说明.md完整算法）
        self.complete_tactical_system = CompleteTacticalSystem(my_intent=my_intent_str)
        logging.info(f"✅ 完整智能战术选择系统已启动 (我方意图: {my_intent_str})")
        
        # 前后攻击队形状态
        self.formation_established = False
        self.wingman_crank_state = {
            "active": False,
            "phase": None,
            "start_time": 0,
            "completed": False,
            "completed_time": 0
        }
        
        # 目标：一字型队形
        # - 僚机在长机后方：dx < 0
        # - 横向对齐：|dy| < 500m
        # - 纵向距离：3-6海里（约5.5-11km）
        self.TARGET_LONGITUDINAL_MIN = 5000  # 最小纵向距离5km
        self.TARGET_LONGITUDINAL_MAX = 11000  # 最大纵向距离11km
        self.TARGET_LATERAL_MAX = 500  # 最大横向偏差500m
        
        # 初始化内部状态
        self.initial_heading = {}
        self.initial_altitude = {}
        self._inner_rnn_states = {}
        
        # 初始化数据记录器
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = get_root_dir() + '/scripts/tacticalProject/tactical_simulation_results'
        self.data_logger = DataLogger(output_dir, timestamp)
        logging.info("✅ 数据记录器初始化完成")
        
        # 初始化雷达系统
        if RADAR_AVAILABLE:
            try:
                self.radar_manager = UnifiedRadarManager()
                logging.info("✅ 雷达系统初始化完成 (APG-68 vs N001VE)")
                logging.info("   我方: F-16C AN/APG-68(V)9 雷达")
                logging.info("   敌方: Su-27 N001VE 雷达")
            except Exception as e:
                logging.warning(f"雷达系统初始化失败: {e}，将使用简化模拟")
                self.radar_manager = None
        else:
            self.radar_manager = None
            logging.warning("⚠️ 雷达系统未集成，使用简化模拟")
        
        # 加载baseline模型用于底层控制
        self._load_baseline_model()
        
        # ===== CAP区域约束 - 跑马道形状（南北向椭圆） =====
        # 定义CAP（Combat Air Patrol）巡逻区域：类似400米跑道
        # 【P0修复】：大幅扩大区域，避免战术机动时频繁越界
        # 配置：长轴240km（南北/X方向），短轴120km（东西/Y方向）
        # 红方初始位置：X=-60km（南端），蓝方：X=+60km（北端）
        # 两端半圆在南北两端，直道沿南北分布
        self.cap_boundary = {
            'type': 'racetrack',  # 跑马道类型
            'center_x': 0,        # 中心X坐标
            'center_y': 0,        # 中心Y坐标
            'length': 240000,     # 长轴240km（原140km）
            'width': 120000,      # 短轴120km（原70km）
            'z_min': 3000,        # 3km最低高度（原5km）
            'z_max': 18000,       # 18km最高高度（原15km）
            # 计算矩形部分和半圆部分（南北向）
            'rect_x_min': -60000,  # 矩形南边界 -60km（原-35km）
            'rect_x_max': 60000,   # 矩形北边界 +60km（原+35km）
            'rect_y_min': -60000,  # 矩形西边界 -60km（原-35km）
            'rect_y_max': 60000,   # 矩形东边界 +60km（原+35km）
            'circle_radius': 60000, # 半圆半径 60km（原35km）
            'south_circle_center': (-60000, 0),   # 南端半圆圆心（X=-60km）
            'north_circle_center': (60000, 0),    # 北端半圆圆心（X=+60km）
        }
        logging.info(f"✅ CAP区域设置: 南北向跑马道（已扩大）")
        logging.info(f"   长轴: 240km (南北/X方向)")
        logging.info(f"   短轴: 120km (东西/Y方向)")
        logging.info(f"   矩形部分: X=[-60,+60]km Y=[-60,+60]km")
        logging.info(f"   南端半圆: 圆心X=-60km, 半径60km")
        logging.info(f"   北端半圆: 圆心X=+60km, 半径60km")
        logging.info(f"   高度: Z=[3,18]km")
        
        # ===== 战术模板核心数据结构 =====
        # 高度指令数组 (15个选项，索引0-14)
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0  # 索引7 = 0m变化（平稳飞行）
        
        # 航向指令数组 (17个选项，索引0-16)
        self.norm_delta_heading = np.array([
            -np.pi,           # 索引0:  -180°
            -2*np.pi/3,       # 索引1:  -120°
            -np.pi/2,         # 索引2:  -90°
            -5*np.pi/12,      # 索引3:  -75°
            -np.pi/3,         # 索引4:  -60°
            -np.pi/4,         # 索引5:  -45°
            -np.pi/6,         # 索引6:  -30°
            -np.pi/12,        # 索引7:  -15°
            0,                # 索引8:  0° (平稳飞行)
            np.pi/12,         # 索引9:  15°
            np.pi/6,          # 索引10: 30°
            np.pi/4,          # 索引11: 45°
            np.pi/3,          # 索引12: 60°
            5*np.pi/12,       # 索引13: 75°
            np.pi/2,          # 索引14: 90°
            2*np.pi/3,        # 索引15: 120°
            np.pi             # 索引16: 180°
        ])
        
        # 速度指令数组 (7个选项，索引0-6)
        self.norm_delta_velocity = np.array([
            -150, -100, -50, 0, 50, 100, 150
        ]) / 100.0  # 索引3 = 0m/s变化（平稳飞行）
        
        # RNN隐藏状态管理
        self._inner_rnn_states = {}
        
        # ===== 战术执行系统 =====
        # 战术距离节点（范围阶段边界）- 精确定义
        # 统一使用constants中的命名
        self.tactical_distances = {
            'MELD': CONTROL_RANGES['MELD'] * 1000,      # 100km: 雷达融合&调整编队
            'MTR': CONTROL_RANGES['MTR'] * 1000,        # 80km: MTR1 跟踪目标&进入作战姿态
            'LR': CONTROL_RANGES['LR'] * 1000,          # 78km: LR1 导弹最大发射距离
            'TR': CONTROL_RANGES['TR'] * 1000,          # 75km: TR1 中制导结束
            'DOR': CONTROL_RANGES['DOR'] * 1000,        # 70km: Beam/short skate/Notch back
            'DR': CONTROL_RANGES['DR'] * 1000,          # 65km: 重新转入进攻/脱离
            'MTR_PRIME': CONTROL_RANGES['MTR_PRIME'] * 1000,  # 55km: 第二次进攻-跟踪目标
            'LR_PRIME': CONTROL_RANGES['LR_PRIME'] * 1000,    # 53km: 第二次进攻-发射
            'TR_PRIME': CONTROL_RANGES['TR_PRIME'] * 1000,    # 50km: 第二次进攻-规避
            'MAR': CONTROL_RANGES['MAR'] * 1000,        # 40km: 最小规避距离，强制脱离
        }
        
        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        
        # 导弹发射状态追踪
        self.missile_launched = {}  # {agent_id: bool}
        
        # 机动状态追踪
        self.maneuver_states = {}  # 存储各飞机的机动状态
        self.short_skate_states = {}  # Short Skate专用状态（原战术模板）
        self.short_skate_start_time = {}  # Short Skate开始时间
        self.initial_heading = {}  # 初始航向
        self.initial_altitude = {}  # 初始高度
        
        # DR节点时间窗口追踪（文档要求：20s窗+5NM侧对保持）
        self.dr_start_time = None  # DR阶段开始时间
        self.dr_beam_completed = False  # 是否完成Beam保持
        self.dr_time_window = 20.0  # 20秒时间窗口
        
        # 二次进攻标记与LR发射轮次追踪
        self.is_second_attack = False  # 是否进入第二轮进攻
        self.lr_launch_count = {}  # {agent_id: int} 记录各机在LR节点的发射次数
        
        # ===== 决策系统 =====
        self.situation_evaluator = SituationEvaluator()  # 统一5维评估器
        self.threat_evaluator = ThreatEvaluator()  # 保留兼容性
        self.intent_predictor = IntentPredictor()
        self.decision_table = DecisionTable()
        self.tactic_selector = TacticSelectorV2(self.threat_evaluator)
        
        # 我方意图（可配置）
        self.my_intent = my_intent_str  # 与配置一致：AGGRESSIVE_CLEAR, CONSERVATIVE_CLEAR, DEFENSIVE
        
        # 当前选定的战术（由决策系统决定）
        self.selected_tactic = None
        self.tactical_roles = {'lead': 'lead', 'wingman': 'wingman'}  # 默认角色
        self.formation_commands = {}  # {agent_id: (alt_id, hdg_id, vel_id)}
        self.vertical_split_targets = {}  # {agent_id: target_altitude} 上下夹击目标高度
        self.enemy_escape_confirm = 0  # 敌机逃逸意图连续确认计数
        
        # 决策节点标记
        self.last_decision_phase = None
        
        # 长机/僚机独立阶段追踪
        self.agent_phases = {}  # {agent_id: TacticalPhase}
        
        # 僚机时间线滞后设置
        self.wingman_delay = {
            TacticalPhase.TR_DOR: 4000,    # 僚机TR_DOR阶段滞后4km
            TacticalPhase.DOR_DR: 8000,    # DOR_DR阶段滞后8km
        }
        
        # ===== 敌方AI系统 =====
        self.enemy_ai = EnemyAIAdapter(project_name="tactical_simulation")
        logging.info("✅ 敌方AI系统已启用")
        
        # ===== 导弹系统 =====
        self.missiles = {}  # {missile_uid: missile_simulator}
        self.last_missile_launch_time = {}  # {agent_id: time}
        self.missile_cooldown = 10.0  # 导弹发射冷却时间（10秒）
        
        logging.info("✅ 战术模板系统初始化完成（归一化数组 + RNN状态 + 战术执行）")
        logging.info("✅ 敌方AI系统初始化完成")
        logging.info("✅ 导弹系统初始化完成")
        logging.info("TacticalTask初始化完成")
    
    def _load_baseline_model(self):
        """加载baseline模型 - 完全按照战术模板实现"""
        try:
            # 创建BaselineActor
            self.baseline_actor = BaselineActor(input_dim=12)
            
            # 加载权重
            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            
            self.baseline_actor.load_state_dict(checkpoint)
            self.baseline_actor.eval()
            logging.info(f"✅ Baseline模型加载成功: {model_path}")
        except Exception as e:
            logging.error(f"Baseline模型加载失败: {e}")
            self.baseline_actor = None
    
    def reset(self, env):
        """重置任务"""
        # 重置内部状态
        self.initial_heading = {}
        self.initial_altitude = {}
        self.missile_launched = {}
        self.maneuver_states = {}
        self.short_skate_states = {}
        self.short_skate_start_time = {}
        self.agent_phases = {}
        self.formation_commands = {}
        self.vertical_split_targets = {}
        self.enemy_escape_confirm = 0
        self.next_round_tactic = None
        self.dr_start_time = None
        self.dr_beam_completed = False
        self.is_second_attack = False
        self.lr_launch_count = {}
        
        # 重置敌方AI系统
        if hasattr(self, 'enemy_ai'):
            self.enemy_ai.reset_all_agents()
        
        # 重置导弹系统
        self.missiles = {}
        self.last_missile_launch_time = {}
        
        # 重置前后攻击位置调整标记
        self.front_back_position_adjusted = False
        
        # 输出初始位置信息
        logging.info("=" * 80)
        logging.info("🔧 [环境重置] 初始位置信息")
        for agent_id in ["A0100", "A0200", "B0100", "B0200"]:
            aircraft = env._jsbsims.get(agent_id)
            if aircraft:
                pos = aircraft.get_position()
                pos_km = pos / 1000
                logging.info(f"   {agent_id}: 位置=({pos_km[0]:.1f}, {pos_km[1]:.1f}, {pos_km[2]:.1f})km")
        
        # 计算初始距离
        distance = self._calculate_distance(env)
        logging.info(f"   ⚠️ 初始距离: {distance/1000:.1f}km")
        if distance < 120000:
            logging.warning(f"   ⚠️ 警告：初始距离({distance/1000:.1f}km) < NLT距离(120km)，将跳过战术阶段！")
        logging.info("=" * 80)
        
        logging.info("TacticalTask重置完成")
        return super().reset(env)
    
    def step(self, env):
        """执行一步仿真 - 核心方法"""
        current_time = env.current_step * env.time_interval
        
        # 更新雷达系统（每步都更新）
        if self.radar_manager:
            try:
                self._update_radar_system(env, current_time)
            except Exception as e:
                logging.warning(f"雷达系统更新失败: {e}")
        
        # 记录数据（每10步记录一次，减少开销）
        if env.current_step % 10 == 0:
            self._record_simulation_data(env, current_time)
        
        # 更新导弹状态
        self._update_missiles(env, current_time)
        
        # 每120秒打印一次详细调试信息
        if env.current_step % 600 == 0:
            logging.info(f"\n{'='*60}")
            # 修复问题4：减少详细状态打印频率（从10秒改为30秒）
            if env.current_step % 360 == 0:  # 360步 ≈ 30秒
                # 打印飞机位置和速度（添加时间帧）
                for agent_id in env.agents.keys():
                    if env.agents[agent_id].is_alive:
                        pos = env.agents[agent_id].get_position()
                        vel = env.agents[agent_id].get_velocity()
                        speed = np.linalg.norm(vel)
                        alt = pos[2]
                        logging.info(f"[T={current_time:.1f}s][{agent_id}] 位置: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}m) "
                                   f"速度: {speed:.1f}m/s 高度: {alt:.1f}m")
            logging.info(f"{'='*60}\n")
        
        # ===== 战术模板架构：动作由normalize_action生成 =====
        # 注意：不在这里生成动作！动作由环境调用normalize_action生成
        # 这里只负责获取观测、奖励、终止条件
        
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
            # 简化奖励计算
            reward_dict[agent_id] = 1.0 if env.agents[agent_id].is_alive else -100.0
        
        # 处理导弹发射（在所有agent处理完后）
        self._handle_missile_launches(env, current_time)
        
        # 检查是否结束，如果结束则保存数据
        if any(done_dict.values()):
            self.data_logger.save_all()
        
        # 环境期望字典格式
        return obs_dict, obs_dict, reward_dict, done_dict, info_dict
    
    
    def _use_lowlevel_policy(self, env, agent_id: str, altitude_cmd_id: int, heading_cmd_id: int, velocity_cmd_id: int) -> np.ndarray:
        """
        使用baseline模型生成底层控制
        完全按照drag_shoot_tactical_task.py实现
        """
        if self.baseline_actor is None:
            # 没有baseline模型，使用简单控制
            # 顺序：[副翼, 升降舵, 方向舵, 油门]
            return np.array([0.0, 0.0, 0.0, 0.7])
        
        try:
            # 1. 获取原始观测
            raw_obs = self.get_obs(env, agent_id)
            
            # 2. 构造12维输入：[norm_alt, norm_heading, norm_vel, obs[:9]]
            input_obs = np.zeros(12)
            
            # 安全索引访问，防止越界
            altitude_cmd_id = min(altitude_cmd_id, len(self.norm_delta_altitude) - 1)
            heading_cmd_id = min(heading_cmd_id, len(self.norm_delta_heading) - 1)
            velocity_cmd_id = min(velocity_cmd_id, len(self.norm_delta_velocity) - 1)
            
            input_obs[0] = self.norm_delta_altitude[altitude_cmd_id]
            input_obs[1] = self.norm_delta_heading[heading_cmd_id]
            input_obs[2] = self.norm_delta_velocity[velocity_cmd_id]
            input_obs[3:12] = raw_obs[:9]
            input_obs = np.nan_to_num(input_obs, nan=0.0)
            
            # 调试：打印输入给baseline的指令（已注释）
            # if env.current_step % 60 == 0 and agent_id.startswith('A'):
            #     logging.info(f"   → baseline输入: 归一化值 alt={input_obs[0]:.3f} hdg={input_obs[1]:.3f} vel={input_obs[2]:.3f}")
            
            input_obs = np.expand_dims(input_obs, axis=0)
            
            # 3. 初始化RNN状态
            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))
            
            # 4. 模型推理
            _action, _rnn_states = self.baseline_actor(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()
            
            # 5. 归一化输出 - 关键转换公式
            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.   # 副翼
            norm_act[1] = action_output[1] / 20 - 1.   # 升降舵
            norm_act[2] = action_output[2] / 20 - 1.   # 方向舵
            norm_act[3] = action_output[3] / 58 + 0.4  # 油门
            
            # 调试：打印baseline的输出（每30步）
            # if env.current_step % 30 == 0:
            #     logging.warning(f"📤 [baseline输出] {agent_id}")
            #     logging.warning(f"   原始输出: {action_output}")
            #     logging.warning(f"   归一化: 副翼={norm_act[0]:.3f} 升降={norm_act[1]:.3f} 方向={norm_act[2]:.3f} 油门={norm_act[3]:.3f}")
            
            # 6. 安全检查：低高度时避免下降（提高阈值避免撞地）
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 3000:
                # ✅ 修复：提高保护阈值到3000m，避免baseline模型导致撞地
                # 原因：敌方AI baseline模型可能产生过度俯冲指令
                if current_alt < 1500:
                    norm_act[1] = max(norm_act[1], 0.5)  # 强制大幅上拉
                    norm_act[3] = 1.0  # 最大油门
                    if env.current_step % 60 == 0:
                        logging.warning(f"🛡️ [{agent_id}] 紧急拉升: 高度{current_alt:.0f}m < 1500m")
                else:
                    norm_act[1] = max(norm_act[1], 0.0)  # 不允许下降
                    norm_act[3] = max(norm_act[3], 0.8)  # 增加油门
                    if env.current_step % 120 == 0:
                        logging.info(f"🛡️ [{agent_id}] 高度保护: 高度{current_alt:.0f}m < 3000m")
            
            return norm_act
            
        except Exception as e:
            logging.error(f"[{agent_id}] Baseline模型执行错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])
    
    def _update_missiles(self, env, current_time: float):
        """更新导弹状态"""
        # 移除已完成的导弹
        finished_missiles = []
        for missile_uid, missile in self.missiles.items():
            if missile.is_done:
                if missile.is_success:
                    logging.info(f"✅ 导弹 {missile_uid} 命中目标！")
                else:
                    logging.info(f"❌ 导弹 {missile_uid} 未命中")
                finished_missiles.append(missile_uid)
        
        for missile_uid in finished_missiles:
            del self.missiles[missile_uid]
        
        # 每40秒打印一次导弹状态
        if env.current_step % 200 == 0 and self.missiles:
            logging.info(f"🚀 [导弹状态] 当前飞行中的导弹: {len(self.missiles)}枚")
            for missile_uid, missile in self.missiles.items():
                if missile.target_aircraft:
                    target_id = missile.target_aircraft.uid
                    missile_pos = missile.get_position()
                    target_pos = missile.target_aircraft.get_position()
                    distance = np.linalg.norm(np.array(missile_pos) - np.array(target_pos))
                    logging.info(f"  {missile_uid} -> {target_id}: 距离={distance/1000:.1f}km")
    
    def _handle_missile_launches(self, env, current_time: float):
        """处理导弹发射逻辑"""
        # 遍历所有标记为需要发射导弹的agent
        for agent_id in list(self.missile_launched.keys()):
            if not self.missile_launched[agent_id]:
                continue
            
            # 检查agent是否还活着
            if agent_id not in env.agents or not env.agents[agent_id].is_alive:
                self.missile_launched[agent_id] = False
                continue
            
            # 检查冷却时间
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if current_time - last_launch < self.missile_cooldown:
                continue
            
            # 检查导弹数量
            if env.agents[agent_id].num_missiles <= 0:
                logging.warning(f"⚠️ {agent_id} 导弹已用尽")
                self.missile_launched[agent_id] = False
                continue
            
            # 寻找目标（优先固定目标，其次最近敌机）
            from core.target_assignment import get_fixed_target, get_target_with_fallback
            target_id = get_target_with_fallback(agent_id, env)
            target = env.agents.get(target_id) if target_id in env.agents else None
            
            if target is None or not target.is_alive:
                logging.warning(f"⚠️ {agent_id} 未找到有效目标 (目标ID:{target_id})")
                # 不立即标记为False，等待下一轮检查
                continue
            
            # ✅ 关键修复3：检查航向是否朝向敌机（防止背对发射）
            my_pos = env.agents[agent_id].get_position()
            target_pos = target.get_position()
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 计算目标方位角
            delta_x = target_pos[0] - my_pos[0]
            delta_y = target_pos[1] - my_pos[1]
            target_bearing = (np.rad2deg(np.arctan2(delta_y, delta_x)) + 360) % 360
            
            # 计算航向与目标方位的夹角
            heading_error = abs(self._normalize_angle_diff(target_bearing - current_heading))
            
            # 只有朝向敌机±90度范围内才允许发射
            if heading_error > 90:
                if env.current_step % 60 == 0:
                    logging.warning(f"⚠️ [{agent_id}] 发射条件不满足: 航向{current_heading:.0f}°, 目标方位{target_bearing:.0f}°, 夹角{heading_error:.0f}° >90°")
                # 不立即标记为False，等待调整航向
                continue
            
            # 发射导弹
            self._launch_missile(env, agent_id, target, current_time)
            
            # 标记为已发射（防止重复发射）
            self.missile_launched[agent_id] = False
    
    def _find_best_target(self, env, agent_id: str):
        """寻找最佳攻击目标"""
        best_target = None
        min_distance = float('inf')
        
        # 确定敌方阵营
        enemy_prefix = 'B' if agent_id.startswith('A') else 'A'
        
        for target_id, target_agent in env.agents.items():
            if target_id.startswith(enemy_prefix) and target_agent.is_alive:
                current_pos = env.agents[agent_id].get_position()
                target_pos = target_agent.get_position()
                distance = np.linalg.norm(np.array(current_pos) - np.array(target_pos))
                
                if distance < min_distance:
                    min_distance = distance
                    best_target = target_agent
        
        return best_target
    
    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹"""
        try:
            from envs.JSBSim.core.simulatior import MissileSimulator
            
            aircraft = env.agents[agent_id]
            
            # 创建导弹ID
            missile_count = 2 - aircraft.num_missiles + 1
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count}"  # A100 → A1001
            
            # 创建导弹模拟器
            missile = MissileSimulator.create(
                parent=aircraft,
                target=target,
                uid=missile_uid
            )
            
            # 添加到环境
            env.add_temp_simulator(missile)
            
            # 记录导弹
            self.missiles[missile_uid] = missile
            
            # 更新发射时间
            self.last_missile_launch_time[agent_id] = current_time
            
            # 减少导弹数量
            aircraft.num_missiles -= 1
            
            logging.info(f"🚀 {agent_id} 发射导弹 {missile_uid} 攻击 {target.uid}")
            
        except Exception as e:
            logging.error(f"导弹发射失败 {agent_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _update_radar_system(self, env, current_time: float):
        """
        更新雷达系统状态
        包括：探测、跟踪、锁定、电子战效果
        """
        if not self.radar_manager:
            return
        
        try:
            # 更新我方雷达
            self.radar_manager.update_friendly_radar_states(env, current_time)
            # 更新敌方雷达
            self.radar_manager.update_enemy_radar_states(env, current_time)
            
            # 每60步打印一次雷达状态（已注释）
            # if env.current_step % 60 == 0:
            #     for agent_id in ["A0100", "A0200"]:
            #         if agent_id in env.agents and env.agents[agent_id].is_alive:
            #             status = self.radar_manager.friendly_radar_states.get(agent_id, "N/A")
            #             targets = self.radar_manager.friendly_radar_targets.get(agent_id, {})
            #             logging.info(f"[雷达] {agent_id} 状态:{status} 跟踪目标数:{len(targets)}")
        except Exception as e:
            logging.warning(f"雷达系统更新异常: {e}")
    
    def _calculate_distance(self, env) -> float:
        """计算敌我距离（长机到敌方长机）"""
        try:
            # ✅ 关键修复1：保存上一次有效距离，避免敌机被击落后返回inf导致阶段错乱
            if not hasattr(self, 'last_valid_distance'):
                self.last_valid_distance = 120000  # 初始距离120km
            
            leader_red = env._jsbsims.get("A0100")
            leader_blue = env._jsbsims.get("B0100")
            
            if not leader_red or not leader_red.is_alive:
                logging.warning(f"⚠️ 我方长机不可用")
                return self.last_valid_distance
            
            if not leader_blue or not leader_blue.is_alive:
                # 敌方长机被击落，返回上一次有效距离而不是inf
                if env.current_step % 60 == 0:
                    logging.info(f"⚠️ 敌方长机被击落，保持距离{self.last_valid_distance/1000:.1f}km（避免阶段错乱）")
                return self.last_valid_distance
            
            pos_red = np.array(leader_red.get_position())
            pos_blue = np.array(leader_blue.get_position())
            distance = np.linalg.norm(pos_red - pos_blue)
            
            # 保存有效距离
            self.last_valid_distance = distance
            return distance
        except Exception as e:
            logging.warning(f"计算距离失败: {e}")
            return self.last_valid_distance if hasattr(self, 'last_valid_distance') else 120000
    
    def _calculate_distance_between(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离"""
        try:
            pos1 = np.array(aircraft1.get_position())
            pos2 = np.array(aircraft2.get_position())
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"计算距离错误: {e}")
            return float('inf')
    
    def _get_enemy_intent_type(self, env, my_agent_id: str) -> str:
        try:
            my_aircraft = env.agents.get(my_agent_id)
            # 使用固定目标分配：A0100→B0100, A0200→B0200
            enemy_id = get_target_with_fallback(my_agent_id, env)
            enemy_aircraft = env.agents.get(enemy_id)
            if my_aircraft and enemy_aircraft:
                enemy_intent = self.intent_predictor.predict_enemy_intent(enemy_aircraft, my_aircraft, env)
                return self.intent_predictor.classify_intent_type(enemy_intent)
        except Exception as _:
            pass
        return 'NEUTRAL_TYPE'
    
    def _set_ecm(self, agent_id: str, active: bool):
        if not hasattr(self, 'radar_manager') or not self.radar_manager:
            return
        try:
            if hasattr(self.radar_manager, 'set_ecm'):
                self.radar_manager.set_ecm(agent_id, active)
            elif hasattr(self.radar_manager, 'activate_ecm'):
                self.radar_manager.activate_ecm(agent_id, active)
            elif hasattr(self.radar_manager, 'set_ecm_state'):
                self.radar_manager.set_ecm_state(agent_id, active)
        except Exception as _:
            pass
    
    def _update_tactical_phase(self, env, agent_id=None):
        """
        更新战术阶段 - 支持长机/僚机独立阶段
        
        Args:
            agent_id: 如果指定，则更新该飞机的独立阶段；否则更新全局阶段
        """
        distance = self._calculate_distance(env)
        adjusted_distance = distance  # 初始化为实际距离
        
        # 如果指定了agent_id，且是敌方，直接返回（敌方不使用我方战术系统）
        if agent_id and agent_id.startswith('B'):
            return
        
        # 如果指定了agent_id，考虑僚机滞后
        if agent_id:
            is_lead = agent_id.endswith('100')
            
            # 僚机使用固定滞后，避免频繁震荡
            # 拖曳射击、前后攻击、并排射击都需要僚机滞后
            if not is_lead and self.selected_tactic in ['DRAG_SHOOT', 'FRONT_BACK', 'SIDE_BY_SIDE']:
                # 并排射击：僚机在TR-DOR和DOR-DR阶段滞后
                if self.selected_tactic == 'SIDE_BY_SIDE':
                    # 僚机在返航阶段滞后8km（确保长机先返航）
                    adjusted_distance += 8000
                else:
                    adjusted_distance += 4000  # 其他战术固定滞后4km
            
            new_phase = self._determine_phase_by_distance(adjusted_distance)
            
            # 更新该飞机的阶段（添加滞后避免频繁切换）
            old_phase = self.agent_phases.get(agent_id, None)
            
            # 首次初始化：如果是NLT阶段且是长机，立即执行NLT决策
            if old_phase is None and new_phase == TacticalPhase.NLT_MELD:
                if agent_id.endswith('100'):  # 长机
                    logging.info(f"⚡ 我方{agent_id}: 首次进入{new_phase.value}阶段 (距离{adjusted_distance/1000:.0f}km)")
                    self.agent_phases[agent_id] = new_phase
                    self._make_decision_at_phase(env, agent_id, new_phase)
                else:
                    self.agent_phases[agent_id] = new_phase
            elif new_phase != old_phase:
                # ✅ 关键修复3：禁止阶段回退 - 战斗已开始拒绝回退到NLT_MELD
                if new_phase == TacticalPhase.NLT_MELD and old_phase not in [None, TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
                    logging.warning(f"⚠️ [{agent_id}] 战斗已开始({old_phase.value})，拒绝回退到NLT_MELD（距离{adjusted_distance/1000:.1f}km）")
                    return  # 保持当前阶段不变，不执行NLT决策
                
                current_time = env.current_step * env.time_interval
                # 只在重要阶段转换时打印（简化）
                if new_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR, TacticalPhase.TR_DOR]:
                    logging.info(f"⚡ 我方{agent_id}: {new_phase.value}阶段 (距离{adjusted_distance/1000:.0f}km)")
                self.agent_phases[agent_id] = new_phase
                
                # 在决策节点执行决策
                self._make_decision_at_phase(env, agent_id, new_phase)
        else:
            # 更新全局阶段（用于整体态势判断）
            new_phase = self._determine_phase_by_distance(distance)
            
            if new_phase != self.current_phase:
                current_time = env.current_step * env.time_interval
                # 只在重要阶段转换时打印（简化）
                if new_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR, TacticalPhase.LR_TR, TacticalPhase.TR_DOR]:
                    logging.info(f"⚡ 全局: {new_phase.value}阶段 (距离{distance/1000:.0f}km)")
                self.current_phase = new_phase
    
    def _determine_phase_by_distance(self, distance):
        """
        根据距离确定阶段
        支持第二次进攻：当距离从DR增大回到MTR区间时，识别为MTR2/LR2/TR2
        """
        # 判断是否在第二次进攻（距离曾经到达过DR，现在又拉大）
        if not hasattr(self, 'min_distance_reached'):
            self.min_distance_reached = float('inf')
            # 注意：不再使用self.second_attack，统一使用self.is_second_attack
        
        # 更新最小距离
        if distance < self.min_distance_reached:
            self.min_distance_reached = distance
        
        # 如果距离曾经到达过DR（<65km），现在又拉大到MTR以上（>80km），说明进入第二次进攻
        if self.min_distance_reached < self.tactical_distances['DR'] and distance > self.tactical_distances['MTR']:
            if not self.is_second_attack:
                # 只在首次进入第二次进攻时输出日志
                self.is_second_attack = True
                logging.info(f"🔄 [第二轮攻击] 距离从{self.min_distance_reached/1000:.1f}km拉大到{distance/1000:.1f}km，进入MTR2阶段")
        
        # ✅ 关键修复2：阶段异常保护 - 距离异常时保持当前阶段
        if distance > 200000 or distance == float('inf'):
            # 距离异常（如inf或>200km），保持当前阶段避免错乱
            if env.current_step % 60 == 0:
                logging.warning(f"⚠️ 距离异常({distance/1000:.1f}km)，保持当前阶段")
            return self.current_phase if hasattr(self, 'current_phase') else TacticalPhase.NLT_MELD
        
        # 阶段判定（根据距离和是否第二轮攻击）
        if distance > self.tactical_distances['MELD']:  # > 100km
            phase = TacticalPhase.NLT_MELD
        elif distance > self.tactical_distances['MTR']:  # > 80km and <= 100km
            phase = TacticalPhase.MELD_MTR
        elif distance > self.tactical_distances['LR']:  # > 78km and <= 80km
            # MTR阶段：区分第一轮和第二轮
            if self.is_second_attack:
                logging.debug(f"   [MTR2阶段] 距离{distance/1000:.1f}km")
            phase = TacticalPhase.MTR_LR
        elif distance > self.tactical_distances['TR']:  # > 75km and <= 78km
            # LR阶段：区分第一轮和第二轮
            if self.is_second_attack:
                logging.debug(f"   [LR2阶段] 距离{distance/1000:.1f}km")
            phase = TacticalPhase.LR_TR
        elif distance > self.tactical_distances['DOR']:  # > 70km and <= 75km
            # TR阶段：区分第一轮和第二轮
            if self.is_second_attack:
                logging.debug(f"   [TR2阶段] 距离{distance/1000:.1f}km")
            phase = TacticalPhase.TR_DOR
        elif distance > self.tactical_distances['DR']:  # > 65km and <= 70km
            phase = TacticalPhase.DOR_DR
        elif distance > self.tactical_distances['MAR']:  # > 40km and <= 65km
            phase = TacticalPhase.DR_MAR
        else:  # <= 40km
            phase = TacticalPhase.BEYOND_MAR
        
        return phase
    
    def _make_decision_at_phase(self, env, agent_id, phase):
        """
        在阶段转换时执行决策
        
        决策节点：
        - NLT_MELD: 策略决策 + 战术决策
        - MELD_MTR: 战术决策（调整）
        - MTR_LR: 机动决策（是否撤退）
        - TR_DOR: 机动决策（是否撤退）
        - DOR_DR: 战术决策（规避+预决策下一轮）
        - DR_MAR: 战术决策（重新进攻或返航）
        """
        # 只在长机时执行战术级决策
        is_lead = agent_id.endswith('100')
        
        if phase == TacticalPhase.NLT_MELD and is_lead:
            # NLT节点：策略决策 + 战术决策
            self._decide_at_nlt(env)
        
        elif phase == TacticalPhase.MELD_MTR and is_lead:
            # MELD节点：战术决策（调整）
            self._decide_at_meld(env)
        
        elif phase == TacticalPhase.MTR_LR:
            # MTR节点：机动决策（是否撤退）+ 第二轮战术应用
            if self.is_second_attack and is_lead:
                # MTR2：应用DR节点预选的战术
                self._decide_at_mtr2(env)
            else:
                # MTR1：标准机动决策
                self._decide_at_mtr(env, agent_id)
        
        elif phase == TacticalPhase.LR_TR:
            # LR节点：参数决策（Crank或平飞）+ 第二轮威胁检查
            if self.is_second_attack:
                # LR2：检查威胁等级，决定是否放弃导弹
                self._decide_at_lr2(env, agent_id)
            else:
                # LR1：标准发射决策
                self._decide_at_lr(env, agent_id)
        
        elif phase == TacticalPhase.TR_DOR:
            # TR节点：机动决策（继续进攻或脱离）
            if self.is_second_attack:
                # TR2：第二轮规避评估
                self._decide_at_tr2(env, agent_id)
            else:
                # TR1：标准规避决策
                self._decide_at_tr(env, agent_id)
        
        elif phase == TacticalPhase.DOR_DR and is_lead:
            # DOR节点：战术决策（规避+预决策下一轮）
            self._decide_at_dor(env)
        
        elif phase == TacticalPhase.DR_MAR and is_lead:
            # DR节点：战术决策（重新进攻或返航）
            self._decide_at_dr(env)
        
        elif phase == TacticalPhase.BEYOND_MAR:
            # MAR节点：机动决策（强制脱离）
            self._decide_at_mar(env, agent_id)
    
    def _form_initial_formation(self, env):
        """
        NLT节点：形成初始并排战斗队形
        - 双机同高度
        - 横向间隔1-2km
        - 相对航向对齐
        - 速度同步
        """
        try:
            lead = env.agents.get('A0100')
            wingman = env.agents.get('A0200')
            
            if not (lead and lead.is_alive and wingman and wingman.is_alive):
                return None
            
            # 获取当前状态
            lead_pos = lead.get_position()
            wingman_pos = wingman.get_position()
            lead_heading = lead.get_property_value(c.attitude_psi_deg)
            
            # 目标：僚机在长机右侧1.5km
            target_offset = np.array([0, 1500, 0])  # 右侧1.5km
            target_wingman_pos = lead_pos + target_offset
            
            # 计算僚机需要的调整
            pos_diff = target_wingman_pos - wingman_pos
            distance_error = np.linalg.norm(pos_diff[:2])  # 水平总距离
            altitude_error = pos_diff[2]  # 高度差（Z方向）
            lateral_error = pos_diff[1]  # 横向距离差（Y方向，东西向）
            longitudinal_error = pos_diff[0]  # 纵向距离差（X方向，南北向）
            
            formation_params = {
                'target_lateral_offset': 1500,  # 1.5km横向间隔
                'target_altitude_diff': 0,  # 同高度
                'target_heading': lead_heading,  # 对齐航向
                'distance_error': distance_error,  # 总距离误差
                'altitude_error': altitude_error,  # 高度误差
                'lateral_error': lateral_error,  # 横向误差（Y方向）
                'longitudinal_error': longitudinal_error,  # 纵向误差（X方向）
            }
            
            logging.info(f"📐 [NLT编队成形] 目标：并排队形，横向1.5km，同高度")
            logging.info(f"   当前误差：横向{lateral_error:.0f}m，纵向{longitudinal_error:.0f}m，高度{altitude_error:.0f}m（总距{distance_error:.0f}m）")
            
            return formation_params
            
        except Exception as e:
            logging.error(f"编队成形计算错误: {e}")
            return None
    
    def _adjust_formation_for_tactic(self, env, tactic: str):
        """
        MELD节点：根据战术调整编队
        - 钳形攻势：左右分离角（30-45度）
        - 上下夹击：高度差（1000-2000m）
        - 拖曳射击：纵向间距（2-3km）
        - 前后攻击：纵向间距（1-2km）
        - 并排射击：横向间隔（1-2km）
        """
        try:
            lead = env.agents.get('A0100')
            wingman = env.agents.get('A0200')
            
            if not (lead and lead.is_alive and wingman and wingman.is_alive):
                return None
            
            adjustment_params = {}
            
            if tactic == 'PINCER_ATTACK':
                # 钳形：左右分离30度
                adjustment_params = {
                    'type': 'lateral_split',
                    'heading_offset': 30,  # 度
                    'lateral_offset': 2000,  # 2km
                    'altitude_diff': 0,
                }
                logging.info(f"📐 [MELD编队调整] 钳形攻势：左右分离30度，间隔2km")
                
            elif tactic == 'HIGH_LOW_ATTACK':
                # ✅ 修复1.3: 上下夹击：高度差2500m (符合2-3km定义)
                adjustment_params = {
                    'type': 'vertical_split',
                    'altitude_diff': 2500,  # 2.5km (修改前: 1500)
                    'lateral_offset': 1000,  # 1km
                    'heading_offset': 0,
                }
                logging.info(f"📐 [MELD编队调整] 上下夹击：高度差2500m")

            elif tactic == 'DRAG_SHOOT':
                # 拖曳射击：纵向间距2.5km
                adjustment_params = {
                    'type': 'longitudinal',
                    'longitudinal_offset': 2500,  # 2.5km
                    'lateral_offset': 500,  # 0.5km
                    'altitude_diff': 0,
                }
                logging.info(f"📐 [MELD编队调整] 拖曳射击：纵向间距2.5km")

            elif tactic in ['SEQUENTIAL_ATTACK', 'FRONT_BACK']:
                # ✅ 修复1.4: 前后攻击：纵向间距8km (符合5.5-11km定义)
                adjustment_params = {
                    'type': 'longitudinal',
                    'longitudinal_offset': 8000,  # 8km (修改前: 1500)
                    'lateral_offset': 300,  # 0.3km
                    'altitude_diff': 0,
                }
                logging.info(f"📐 [MELD编队调整] 前后攻击：纵向间距8km")

            else:  # SIDE_BY_SIDE
                # ✅ 修复1.2: 并排射击：横向间隔6km (符合3-5NM=5.5-9.3km定义)
                adjustment_params = {
                    'type': 'side_by_side',
                    'lateral_offset': 6000,  # 6km (修改前: 1500)
                    'altitude_diff': 0,
                    'heading_offset': 0,
                }
                logging.info(f"📐 [MELD编队调整] 并排射击：横向间隔6km")
            
            return adjustment_params
            
        except Exception as e:
            logging.error(f"编队调整计算错误: {e}")
            return None
    
    def _apply_formation_params(self, env, params: dict):
        """
        将编队成形参数转化为指令（针对NLT并排队形）
        目标：僚机在长机右侧1.5km、同高度、航向对齐
        """
        try:
            lead = env.agents.get('A0100')
            wing = env.agents.get('A0200')
            if not (lead and wing and lead.is_alive and wing.is_alive):
                return
            # ✅ 计算位置和误差
            lead_pos = lead.get_position()
            wing_pos = wing.get_position()
            lateral_err = params.get('lateral_error', 0.0)
            longitudinal_err = params.get('longitudinal_error', 0.0)
            target_heading = params.get('target_heading', lead.get_property_value(c.attitude_psi_deg))
            wingman_heading = wing.get_property_value(c.attitude_psi_deg)
            
            # ✅ 航向控制：直接对齐长机航向（完全简化，稳定可靠）
            # 计算航向差（标准化到[-180, 180]）
            heading_diff = self._normalize_angle_diff(target_heading - wingman_heading)
            hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
            
            # 日志输出
            if abs(lateral_err) > 3000:
                logging.info(f"🧭 [编队重组] A0200横向{lateral_err:.0f}m，对齐长机航向{target_heading:.1f}°（差{heading_diff:.1f}°）")
            
            # ✅ 修复：高度对齐 + 完整安全检查
            alt_err = params.get('altitude_error', 0.0)  # 目标位置高度 - 当前僚机高度
            current_wingman_alt = wing.get_position()[2]  # 僚机当前高度（米）
            current_lead_alt = lead.get_position()[2]  # 长机当前高度（米）
            
            # 安全高度范围：4-15km（留出1km缓冲区）
            SAFE_ALT_MIN = 4000  # 4km
            SAFE_ALT_MAX = 15000  # 15km
            ALT_TOLERANCE = 500  # 高度差<500m时不调整
            
            # ✅ 修复：提前计算僚机速度（后续多处使用）
            # 使用地面坐标系速度（NED坐标系），单位已经是m/s
            wingman_speed = np.sqrt(
                wing.get_property_value(c.velocities_v_north_mps)**2 + 
                wing.get_property_value(c.velocities_v_east_mps)**2 + 
                wing.get_property_value(c.velocities_v_down_mps)**2
            )
            
            # ✅ 调试日志：输出速度和高度差，用于验证检测逻辑
            if abs(alt_err) > ALT_TOLERANCE:
                logging.info(f"📊 [编队重组检测] A0200速度={wingman_speed:.1f}m/s, 高度差={alt_err:.0f}m, 当前高度={current_wingman_alt:.0f}m")
            
            # 决策逻辑：✅ 增强低空强制保护
            if current_wingman_alt < 4500:
                # ✅ 低空强制爬升（最高优先级，提高阈值到4.5km）
                alt_idx = 12  # 强制爬升+500m
                logging.warning(f"🚨 [低空保护] A0200高度{current_wingman_alt:.0f}m<4.5km，强制爬升")
            elif abs(alt_err) < ALT_TOLERANCE:
                # 高度差很小，保持不动
                alt_idx = 7
            elif current_wingman_alt > SAFE_ALT_MAX:
                # 当前高度过高（>15km），只能保持或下降
                if alt_err < 0:  # 需要下降
                    alt_idx = 2  # 下降-500m
                    logging.info(f"🛡️ [编队重组] A0200高度{current_wingman_alt:.0f}m过高，下降")
                else:
                    alt_idx = 7  # 保持高度
                    logging.warning(f"🛡️ [编队重组] A0200高度{current_wingman_alt:.0f}m已接近上限，保持高度")
            else:
                # 在安全范围内（4-15km），简化高度控制
                # ✅ 高空强制保护
                if current_wingman_alt > 8500:
                    alt_idx = 2  # 强制下降-500m
                    logging.warning(f"🚨 [高空保护] A0200高度{current_wingman_alt:.0f}m>8.5km，强制下降")
                # ✅ 简化高度调整：直接根据高度差调整
                elif abs(alt_err) < 200:
                    alt_idx = 7  # 保持高度
                elif alt_err > 800:
                    alt_idx = 9  # 爬升+100m（保守）
                elif alt_err < -800:
                    alt_idx = 5  # 下降-100m（保守）
                else:
                    alt_idx = 7  # 保持高度
            
            # ✅ 速度控制：简化策略
            dist_err = params.get('distance_error', 0.0)
            
            # 高空+高速保护
            if current_wingman_alt > 8000 and wingman_speed > 280:
                vel_idx = self._convert_velocity_to_index(-30.0)  # 减速
                logging.warning(f"🚨 [高空高速保护] A0200减速")
            # 失速保护
            elif wingman_speed < 150:
                vel_idx = 4  # 加速+50m/s
                logging.warning(f"🚨 [失速保护] A0200加速")
            # 极高速保护
            elif wingman_speed > 300:
                vel_idx = self._convert_velocity_to_index(-30.0)  # 减速
            # 正常速度调整：根据距离误差
            elif dist_err > 3000:
                vel_idx = self._convert_velocity_to_index(20.0)  # 温和加速
            elif dist_err > 1000:
                vel_idx = self._convert_velocity_to_index(10.0)  # 微调
            elif dist_err < 500:
                vel_idx = 3  # 保持
            else:
                vel_idx = self._convert_velocity_to_index(10.0)  # 默认微调
            
            self.formation_commands['A0200'] = (alt_idx, hdg_idx, vel_idx)
            # ✅ 修复：长机也需要指令，否则会用默认值造成转圈
            # 长机保持当前航向和速度，僚机负责调整到编队位置
            self.formation_commands['A0100'] = (7, 8, 3)  # 保持高度、航向、中速
            
            # 日志输出（增强版）
            if abs(alt_err) > ALT_TOLERANCE or abs(lateral_err) > 1000:
                logging.info(f"✅ [NLT编队执行] 僚机指令 alt_id={alt_idx}（高度差{alt_err:.0f}m）hdg_id={hdg_idx}（横向{lateral_err:.0f}m）vel_id={vel_idx}（速度{wingman_speed:.0f}m/s）")
            else:
                logging.info(f"✅ [NLT编队执行] 僚机指令 alt_id={alt_idx} hdg_id={hdg_idx} vel_id={vel_idx}")
        except Exception as e:
            logging.error(f"编队成形指令计算错误: {e}")
    
    def _apply_adjustment_params(self, env, params: dict, roles: dict):
        """
        将战术编队调整参数转化为指令（针对MELD战术调整）
        不同type对应不同维度的指令偏置

        修复说明：
        - 高度指令应该是"到目标高度的差值"，而不是"固定增量"
        - 避免累积效应导致长机过度爬升失速、僚机过度下降撞地
        """
        try:
            lead = env.agents.get('A0100')
            wing = env.agents.get('A0200')
            if not (lead and wing and lead.is_alive and wing.is_alive):
                return

            t = params.get('type', 'side_by_side')

            if t == 'lateral_split':
                # 左右分离：长机左、僚机右
                # 【修复5】限制偏航角度，避免过度横向飞行超出CAP边界
                lead_hdg_off = -abs(params.get('heading_offset', 30))
                wing_hdg_off = abs(params.get('heading_offset', 30))

                # 【关键修复】限制偏航角度，确保不超出Y=±60km边界
                # 最大偏航角度：±15度（确保主要向北飞行，而非向东西飞行）
                lead_hdg_off = np.clip(lead_hdg_off, -15.0, 15.0)
                wing_hdg_off = np.clip(wing_hdg_off, -15.0, 15.0)

                lead_hdg_idx = self._convert_heading_to_index(np.deg2rad(lead_hdg_off))
                wing_hdg_idx = self._convert_heading_to_index(np.deg2rad(wing_hdg_off))
                self.formation_commands['A0100'] = (7, lead_hdg_idx, 3)
                self.formation_commands['A0200'] = (7, wing_hdg_idx, 3)

                logging.info(f"📐 [MELD横向分离] 限制后偏航角: 长机={lead_hdg_off:.1f}°, 僚机={wing_hdg_off:.1f}°")

            elif t == 'vertical_split':
                # 上下夹击：设置目标高度，由战术执行函数负责爬升控制
                alt_diff = abs(params.get('altitude_diff', 1500))
                lead_role = roles.get('lead', 'lead')

                # 获取当前高度
                lead_alt = lead.get_property_value(c.position_h_sl_m)
                wing_alt = wing.get_property_value(c.position_h_sl_m)

                # 计算平均高度和目标高度
                avg_alt = (lead_alt + wing_alt) / 2
                target_high_alt = avg_alt + alt_diff / 2  # 高位目标
                target_low_alt = avg_alt - alt_diff / 2   # 低位目标

                # 限制目标高度在安全范围内（3km-10km）
                target_high_alt = np.clip(target_high_alt, 3000, 10000)
                target_low_alt = np.clip(target_low_alt, 3000, 10000)

                # 保存目标高度到状态，由战术执行函数使用
                if lead_role == 'high':
                    self.vertical_split_targets['A0100'] = target_high_alt
                    self.vertical_split_targets['A0200'] = target_low_alt
                else:
                    self.vertical_split_targets['A0100'] = target_low_alt
                    self.vertical_split_targets['A0200'] = target_high_alt

                # 调试日志
                logging.info(f"📐 [MELD高度目标设置] 长机目标:{self.vertical_split_targets['A0100']:.0f}m, "
                            f"僚机目标:{self.vertical_split_targets['A0200']:.0f}m")

            elif t == 'longitudinal':
                # 纵向编队：前后分离，前加速、后减速
                lead_role = roles.get('lead', 'lead')
                front_vel = self._convert_velocity_to_index(+50.0)
                rear_vel = self._convert_velocity_to_index(-30.0)
                if lead_role in ['front', 'drag']:
                    self.formation_commands['A0100'] = (7, 8, front_vel)
                    self.formation_commands['A0200'] = (7, 8, rear_vel)
                else:
                    self.formation_commands['A0100'] = (7, 8, rear_vel)
                    self.formation_commands['A0200'] = (7, 8, front_vel)
            else:
                # 并排：航向对齐
                self.formation_commands['A0100'] = (7, 8, 3)
                self.formation_commands['A0200'] = (7, 8, 3)

            logging.info(f"✅ [MELD编队执行] 指令: {self.formation_commands}")
        except Exception as e:
            logging.error(f"编队调整指令计算错误: {e}")
    
    def _decide_at_nlt(self, env):
        """NLT节点决策：策略决策 + 战术决策 + 编队成形（使用完整智能战术选择系统）"""
        try:
            logging.info("=" * 80)
            logging.info("🎯 [NLT节点] 开始战术决策（完整智能系统）")

            # ===== 使用完整智能战术选择系统（Algorithm 6-1） =====

            # 1. 获取飞机列表
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]

            # 2. 调用完整智能战术选择系统
            self.selected_tactic, self.tactical_roles = self.complete_tactical_system.select_tactic(
                control_distance='NLT',
                my_aircraft_list=my_aircraft,
                enemy_aircraft_list=enemy_aircraft,
                env=env
            )

            logging.info(f"   ✅ 选定战术: {self.selected_tactic}")
            logging.info(f"   ✅ 角色分配: 长机={self.tactical_roles.get('lead', 'left')}, 僚机={self.tactical_roles.get('wingman', 'right')}")

            # 3. ✅ 关键修复：检查是否是敌机坠毁触发的重组
            # 如果已经选择过战术且不在初始阶段，跳过编队重组，直接执行战术
            lead_phase = self.agent_phases.get('A0100', TacticalPhase.NLT_MELD)
            wingman_phase = self.agent_phases.get('A0200', TacticalPhase.NLT_MELD)
            
            # 如果任一飞机已经超过MELD阶段，说明战斗已开始，禁止重组
            if lead_phase not in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR] or \
               wingman_phase not in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
                logging.warning(f"⚠️ 战斗已开始(长机:{lead_phase.value},僚机:{wingman_phase.value})，禁止NLT重组！")
                return  # 直接返回，不执行编队重组
            
            # 首次NLT或MELD阶段才执行编队成形
            if lead_phase == TacticalPhase.NLT_MELD and wingman_phase == TacticalPhase.NLT_MELD:
                formation_params = self._form_initial_formation(env)
                if formation_params:
                    if not hasattr(self, 'formation_params'):
                        self.formation_params = {}
                    self.formation_params['NLT'] = formation_params
                    self._apply_formation_params(env, formation_params)
            else:
                logging.info(f"⚠️ MELD阶段，跳过NLT编队成形，直接执行战术调整")

            logging.info("=" * 80)

            # 记录决策日志
            self.data_logger.log_decision(
                env.current_step * env.time_interval, 'A0100', 'NLT_MELD', 'TACTIC',
                self.selected_tactic, None, f"Complete tactical system selection")

        except Exception as e:
            logging.error(f"❌ NLT决策错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.selected_tactic = 'SIDE_BY_SIDE'
    
    def _decide_at_meld(self, env):
        """MELD节点决策：战术调整（使用完整智能战术选择系统）"""
        try:
            # ===== 使用完整智能战术选择系统（Algorithm 6-1） =====

            # 1. 获取飞机列表
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]

            # 2. 调用完整智能战术选择系统
            new_tactic, new_roles = self.complete_tactical_system.select_tactic(
                control_distance='MELD',
                my_aircraft_list=my_aircraft,
                enemy_aircraft_list=enemy_aircraft,
                env=env
            )

            # 3. 检查是否需要调整战术
            if new_tactic != self.selected_tactic:
                logging.info(f"🔄 MELD调整战术: {self.selected_tactic} → {new_tactic}")
                logging.info(f"🔄 MELD角色调整: 长机={new_roles.get('lead', 'left')}, 僚机={new_roles.get('wingman', 'right')}")

                # 记录决策日志
                self.data_logger.log_decision(
                    env.current_step * env.time_interval, 'A0100', 'MELD_MTR', 'TACTIC',
                    new_tactic, None, f"Adjusted from {self.selected_tactic}")

                self.selected_tactic = new_tactic
                self.tactical_roles = new_roles

            # 4. 根据战术调整编队
            adjustment_params = self._adjust_formation_for_tactic(env, self.selected_tactic)
            if adjustment_params:
                # 存储编队调整参数供后续执行
                if not hasattr(self, 'formation_params'):
                    self.formation_params = {}
                self.formation_params['MELD'] = adjustment_params
                self._apply_adjustment_params(env, adjustment_params, getattr(self, 'tactical_roles', {}))

        except Exception as e:
            logging.error(f"MELD决策错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _decide_at_mtr(self, env, agent_id: str):
        """
        MTR节点决策：机动决策（是否撤退）
        核心任务：前往攻击占位点
        决策内容：检查威胁值，决定是否撤退
        """
        try:
            # 1. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            my_threat = threat_info.get('lead_threat', 0.5) if agent_id.endswith('100') else threat_info.get('wingman_threat', 0.5)
            
            # 2. 获取敌机意图（使用固定目标分配）
            my_aircraft = env.agents.get(agent_id)
            enemy_id = get_target_with_fallback(agent_id, env)
            enemy_aircraft = env.agents.get(enemy_id)
            
            if my_aircraft and enemy_aircraft:
                enemy_intent = self.intent_predictor.predict_enemy_intent(enemy_aircraft, my_aircraft, env)
                enemy_intent_type = self.intent_predictor.classify_intent_type(enemy_intent)
            else:
                enemy_intent_type = 'NEUTRAL_TYPE'
            
            # 3. 决策逻辑（根据文档）
            should_retreat = False
            
            # 保守肃清：检查是否需要撤退
            if self.my_intent == 'CONSERVATIVE_CLEAR':
                # 执行撤退：敌方意图为进攻、且总威胁度>0.8
                if enemy_intent_type == 'ATTACK_TYPE' and my_threat > 0.8:
                    should_retreat = True
                    logging.warning(f"⚠️ [MTR决策] {agent_id} 威胁过高({my_threat:.2f})，决定撤退")
            
            # 防御意图：检查是否脱离
            elif self.my_intent == 'DEFENSIVE':
                if enemy_intent_type == 'ESCAPE_TYPE':
                    should_retreat = True
                    logging.info(f"✅ [MTR决策] {agent_id} 敌机逃离，防御任务完成，准备脱离")
            
            # 记录决策结果
            if not hasattr(self, 'retreat_decision'):
                self.retreat_decision = {}
            self.retreat_decision[agent_id] = should_retreat
            
            # 未撤退则前往占位点（落地姿态与速度准备）
            if not should_retreat:
                alt_id, hdg_id, vel_id = self._compute_attack_approach_commands(env, agent_id)
                self.formation_commands[agent_id] = (alt_id, hdg_id, vel_id)
                logging.info(f"🎯 [MTR占位] {agent_id} 指令 alt_id={alt_id} hdg_id={hdg_id} vel_id={vel_id}")
            
        except Exception as e:
            logging.error(f"MTR决策错误: {e}")
    
    def _decide_at_lr(self, env, agent_id: str):
        """
        LR节点决策：参数决策（Crank或平飞）
        核心任务：完成中制导
        决策内容：根据雷达照射需求，决定Crank或平飞
        
        RWR驱动：
        - RWR≥2（TRACK）：优先CRANK并限制偏角，保证照射余量
        - RWR≥3（LOCK/导弹）：强制CRANK+ECM激活
        
        LR_PRIME（第二次LR）：
        - 文档要求：可终止中制导选择返航
        - 条件：威胁过高或RWR≥3
        """
        try:
            # 1. 判断是否为第二次LR（LR_PRIME）
            if agent_id not in self.lr_launch_count:
                self.lr_launch_count[agent_id] = 0
            
            is_lr_prime = self.lr_launch_count[agent_id] >= 1  # 第二次及以上LR
            
            # 2. 获取RWR状态
            rwr_threat_level = 0
            if hasattr(self, 'radar_manager') and self.radar_manager:
                rwr_data = self.radar_manager.rwr_states.get(agent_id, {})
                rwr_threat_level = rwr_data.get('threat_level', 0)
            
            # 3. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            my_threat = threat_info.get('lead_threat', 0.5) if agent_id.endswith('100') else threat_info.get('wingman_threat', 0.5)
            
            # 4. LR_PRIME专用逻辑：终止中制导判断
            if is_lr_prime:
                # 文档要求：为了规避敌方导弹，可以终止中制导选择返航
                should_abort = False
                
                if rwr_threat_level >= 3:
                    should_abort = True
                    logging.warning(f"⚠️ [LR_PRIME终止] {agent_id} RWR威胁等级{rwr_threat_level}（LOCK/导弹），终止中制导+返航")
                elif my_threat > 0.8:
                    should_abort = True
                    logging.warning(f"⚠️ [LR_PRIME终止] {agent_id} 威胁过高({my_threat:.2f})，终止中制导+返航")
                
                if should_abort:
                    # 标记为强制脱离
                    if not hasattr(self, 'forced_retreat'):
                        self.forced_retreat = {}
                    self.forced_retreat[agent_id] = True
                    # 不增加发射次数
                    return
            
            # 5. 增加LR发射次数
            self.lr_launch_count[agent_id] += 1
            
            # 6. 获取敌机方位
            enemy_bearing = self._get_enemy_bearing(env, agent_id)
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 7. 计算航向差
            heading_diff = ((enemy_bearing - current_heading + 180) % 360) - 180
            
            # 8. 决策逻辑（结合RWR）
            if not hasattr(self, 'lr_maneuver'):
                self.lr_maneuver = {}
            
            # RWR≥3：强制CRANK+ECM
            if rwr_threat_level >= 3:
                self.lr_maneuver[agent_id] = 'CRANK'
                self._set_ecm(agent_id, True)
                logging.warning(f"⚠️ [LR决策+RWR] {agent_id} RWR威胁等级{rwr_threat_level}（LOCK/导弹），强制CRANK+ECM")
            # RWR≥2：优先CRANK
            elif rwr_threat_level >= 2:
                self.lr_maneuver[agent_id] = 'CRANK'
                logging.info(f"🎯 [LR决策+RWR] {agent_id} RWR威胁等级{rwr_threat_level}（TRACK），优先CRANK保留余量")
            # RWR<2：按航向差决定
            elif abs(heading_diff) < 15.0:
                self.lr_maneuver[agent_id] = 'LEVEL_FLIGHT'
                logging.debug(f"[LR决策] {agent_id} 航向差{heading_diff:.1f}°，选择平飞")
            else:
                self.lr_maneuver[agent_id] = 'CRANK'
                logging.debug(f"[LR决策] {agent_id} 航向差{heading_diff:.1f}°，选择Crank")
            
            # ECM关闭条件：低威胁（<3）时关闭，避免持续开启
            if rwr_threat_level < 3:
                self._set_ecm(agent_id, False)
            
        except Exception as e:
            logging.error(f"LR决策错误: {e}")
    
    def _decide_at_tr(self, env, agent_id: str):
        """
        TR节点决策：机动决策（继续进攻或脱离）
        核心任务：中制导结束，规避敌方攻击
        决策内容：检查作战目标是否完成，决定继续或脱离
        
        RWR驱动：
        - RWR≥3（LOCK/导弹）：强制规避（优先NOTCH_BACK/SHORT_SKATE）+立即ECM
        - RWR=2（TRACK）：提前BEAM+机会性ECM
        - RWR=1（SEARCH）：仅保留风险余量
        """
        try:
            # 1. 获取RWR状态
            rwr_threat_level = 0
            if hasattr(self, 'radar_manager') and self.radar_manager:
                rwr_data = self.radar_manager.rwr_states.get(agent_id, {})
                rwr_threat_level = rwr_data.get('threat_level', 0)
            
            # 2. 检查是否已发射导弹
            has_launched = agent_id in self.missile_launched
            
            # 3. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            my_threat = threat_info.get('lead_threat', 0.5) if agent_id.endswith('100') else threat_info.get('wingman_threat', 0.5)
            enemy_intent_type = self._get_enemy_intent_type(env, agent_id)
            # 敌机逃逸连续确认计数
            if enemy_intent_type == 'ESCAPE_TYPE':
                self.enemy_escape_confirm = min(getattr(self, 'enemy_escape_confirm', 0) + 1, 2)
            else:
                self.enemy_escape_confirm = 0
            role_key = 'lead' if agent_id.endswith('100') else 'wingman'
            my_role = self.tactical_roles.get(role_key, None)
            if my_role in ['drag', 'front', 'low']:
                my_threat = min(1.0, my_threat + 0.1)
            elif my_role in ['shooter', 'rear', 'high']:
                my_threat = max(0.0, my_threat - 0.1)
            
            # 4. 决策逻辑（结合RWR/敌意图/角色）
            should_continue = True
            
            # RWR≥3：强制规避
            if rwr_threat_level >= 3:
                should_continue = False
                self._set_ecm(agent_id, True)
                logging.warning(f"⚠️ [TR决策+RWR] {agent_id} RWR威胁等级{rwr_threat_level}（LOCK/导弹），强制规避+ECM")
            # RWR=2：提前BEAM
            elif rwr_threat_level == 2:
                logging.info(f"🎯 [TR决策+RWR] {agent_id} RWR威胁等级{rwr_threat_level}（TRACK），提前BEAM规避")
                # 仍按后续逻辑判断，但加权规避倾向
                my_threat = min(1.0, my_threat + 0.2)  # 提升威胁感知
            else:
                # 无高等级威胁，关闭ECM
                self._set_ecm(agent_id, False)
            
            # 激进肃清：始终继续（除非RWR≥3强制规避）
            if self.my_intent == 'AGGRESSIVE_CLEAR' and rwr_threat_level < 3:
                should_continue = True
            
            # 保守肃清：检查威胁
            elif self.my_intent == 'CONSERVATIVE_CLEAR':
                if enemy_intent_type == 'ATTACK_TYPE' and (my_threat > 0.8 or rwr_threat_level >= 3):
                    should_continue = False
                    logging.warning(f"⚠️ [TR决策] {agent_id} 威胁过高({my_threat:.2f})或RWR={rwr_threat_level}，决定脱离")
                
            # 防御意图：已发射导弹则脱离
            elif self.my_intent == 'DEFENSIVE':
                if has_launched or rwr_threat_level >= 3:
                    should_continue = False
                    logging.info(f"✅ [TR决策] {agent_id} 已完成防御任务或RWR威胁高，准备脱离")
            
            # 记录决策结果
            if not hasattr(self, 'continue_decision'):
                self.continue_decision = {}
            self.continue_decision[agent_id] = should_continue
            
        except Exception as e:
            logging.error(f"TR决策错误: {e}")
    
    def _decide_at_dor(self, env):
        """
        DOR节点决策：战术决策（规避+预决策下一轮）
        核心任务：选择规避机动，预决策下一阶段战术
        决策内容：Beam/Short Skate/Notch back
        
        RWR驱动：
        - RWR≥3（LOCK/导弹）：强制SHORT_SKATE+立即ECM
        - RWR=2（TRACK）：优先BEAM+机会性ECM
        - RWR≤1：按威胁值选择
        """
        try:
            rwr_threat_level = 0
            if hasattr(self, 'radar_manager') and self.radar_manager:
                rwr_data = self.radar_manager.rwr_states.get('A0100', {})
                rwr_threat_level = rwr_data.get('threat_level', 0)
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            threat_level = threat_info.get('situation', 'ADVANTAGE')
            enemy_intent_type = self._get_enemy_intent_type(env, 'A0100')
            # 敌机逃逸连续确认计数
            if enemy_intent_type == 'ESCAPE_TYPE':
                self.enemy_escape_confirm = min(getattr(self, 'enemy_escape_confirm', 0) + 1, 2)
            else:
                self.enemy_escape_confirm = 0
            tactics, maneuvers = self.decision_table.query_dor(self.my_intent, threat_level, enemy_intent_type)
            if rwr_threat_level >= 3:
                evasion_maneuver = 'SHORT_SKATE'
                self._set_ecm('A0100', True)
                self._set_ecm('A0200', True)
            elif rwr_threat_level == 2:
                evasion_maneuver = 'BEAM'
            else:
                if lead_threat > 0.7 and 'SHORT_SKATE' in maneuvers:
                    evasion_maneuver = 'SHORT_SKATE'
                elif lead_threat > 0.4 and 'BEAM' in maneuvers:
                    evasion_maneuver = 'BEAM'
                else:
                    evasion_maneuver = maneuvers[0] if maneuvers else 'NOTCH_BACK'
                # 无高等级威胁，关闭ECM
                self._set_ecm('A0100', False)
                self._set_ecm('A0200', False)
            self.evasion_maneuver = evasion_maneuver
            if tactics:
                self.next_round_tactic = self.selected_tactic if self.selected_tactic in tactics else tactics[0]
            else:
                self.next_round_tactic = None
            
        except Exception as e:
            logging.error(f"DOR决策错误: {e}")
    
    def _decide_at_dr(self, env):
        """
        DR节点决策：战术决策（重新进攻或返航）
        核心任务：根据态势决定是否发起下一轮进攻
        决策内容：检查敌机状态、我方状态、意图
        
        文档要求：
        - 20秒时间窗内，保持Beam侧对状态（约5海里 ~9.26km）
        - 在时间窗内完成下一阶段战术决策
        - 满时后决定是否重新转热或返航
        """
        try:
            current_time = env.current_step * env.time_interval
            
            # 1. 初始化DR开始时间
            if self.dr_start_time is None:
                self.dr_start_time = current_time
                self.dr_beam_completed = False
                logging.info(f"🔄 [DR节点] 开始20秒时间窗（{current_time:.1f}s）")
            
            # 2. 计算时间窗进度
            time_in_dr = current_time - self.dr_start_time
            time_remaining = self.dr_time_window - time_in_dr
            
            # 3. 检查敌我距离（5NM约束）
            distance = self._calculate_distance(env)
            distance_nm = distance / 1852.0  # 转换为海里
            target_distance_nm = 5.0  # 目标保持5海里
            
            if time_in_dr < self.dr_time_window:
                if not self.dr_beam_completed:
                    logging.info(f"⏱️ [DR时间窗] 进度: {time_in_dr:.1f}s/{self.dr_time_window}s，距离: {distance_nm:.1f}NM（目标5NM）")
                    if abs(distance_nm - target_distance_nm) < 1.0:
                        pass

                # ===== 使用完整智能战术选择系统（Algorithm 6-1） =====
                my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
                enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]

                next_tactic, _ = self.complete_tactical_system.select_tactic(
                    control_distance='DR',
                    my_aircraft_list=my_aircraft,
                    enemy_aircraft_list=enemy_aircraft,
                    env=env
                )

                if not hasattr(self, 'next_round_tactic') or self.next_round_tactic != next_tactic:
                    self.next_round_tactic = next_tactic
                    logging.info(f"🎯 [DR时间窗] 预选下一轮战术: {next_tactic}")

                return
            
            # 5. 时间窗满：标记完成并进行最终决策
            if not self.dr_beam_completed:
                self.dr_beam_completed = True
                logging.info(f"✅ [DR时间窗] 20秒侧对保持完成，开始最终决策")
            
            # 6. 获取RWR状态
            rwr_threat_level = 0
            if hasattr(self, 'radar_manager') and self.radar_manager:
                rwr_data = self.radar_manager.rwr_states.get('A0100', {})
                rwr_threat_level = rwr_data.get('threat_level', 0)
            
            # 7. 检查敌机状态
            enemy_alive = []
            for enemy_id in ["B0100", "B0200"]:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    enemy_alive.append(enemy_id)
            
            # 8. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            
            # 9. 最终决策逻辑（结合RWR/决策表/防御意图逃逸确认）
            should_reengage = False
            
            # 激进肃清：只要有敌机且RWR<3就继续
            if self.my_intent == 'AGGRESSIVE_CLEAR':
                if len(enemy_alive) > 0 and rwr_threat_level < 3:
                    should_reengage = True
                    logging.info(f"🔥 [DR决策] 激进肃清，继续进攻剩余{len(enemy_alive)}架敌机")
            
            # 保守肃清：检查威胁、RWR和敌机数量
            elif self.my_intent == 'CONSERVATIVE_CLEAR':
                if len(enemy_alive) > 0 and rwr_threat_level < 3 and self.next_round_tactic is not None and lead_threat < 0.6:
                    should_reengage = True
                    logging.info(f"✅ [DR决策] 保守肃清，威胁可控({lead_threat:.2f})/RWR{rwr_threat_level}，继续进攻")
                else:
                    logging.info(f"🚫 [DR决策] 保守肃清，威胁过高({lead_threat:.2f})/RWR{rwr_threat_level}或无敌机，返航")
            
            # 防御意图：当敌机连续两节点为逃逸则不再进攻，否则按激进肃清逻辑
            elif self.my_intent == 'DEFENSIVE':
                enemy_intent_type = self._get_enemy_intent_type(env, 'A0100')
                if enemy_intent_type == 'ESCAPE_TYPE' and getattr(self, 'enemy_escape_confirm', 0) >= 2:
                    should_reengage = False
                    logging.info(f"🛡️ [DR决策] 防御意图：敌机连续逃逸确认，返航")
                else:
                    if len(enemy_alive) > 0 and rwr_threat_level < 3:
                        should_reengage = True
                        logging.info(f"🛡️ [DR决策] 防御意图：敌机未确认逃逸，按激进逻辑转热")
            
            # 10. 记录决策结果
            self.should_reengage = should_reengage
            
            # 11. 如果重新进攻，选择战术并重置DR时间窗
            if should_reengage:
                if hasattr(self, 'next_round_tactic') and self.next_round_tactic:
                    self.selected_tactic = self.next_round_tactic
                else:
                    threat_info2 = self.threat_evaluator.evaluate_situation(env)
                    threat_level2 = threat_info2.get('situation', 'ADVANTAGE')
                    enemy_intent_type2 = self._get_enemy_intent_type(env, 'A0100')
                    tactics2, _ = self.decision_table.query_dr(self.my_intent, threat_level2, enemy_intent_type2)
                    if tactics2:
                        self.selected_tactic = tactics2[0]
                logging.info(f"🎯 [DR决策] 重新进攻，战术: {self.selected_tactic}")
                # 标记为二次进攻
                self.is_second_attack = True
                logging.info(f"🔄 [DR决策] 进入第二轮进攻（MTR_PRIME/LR_PRIME/TR_PRIME）")
                # 重置DR状态以备下一轮
                self.dr_start_time = None
                self.dr_beam_completed = False
            
        except Exception as e:
            logging.error(f"DR决策错误: {e}")
    
    def _decide_at_mtr2(self, env):
        """
        MTR2节点决策：第二轮攻击的战术应用
        核心任务：
        1. 应用DR节点预选的战术
        2. 检查威胁等级决定是否继续
        3. 重新调整队形
        """
        try:
            current_time = env.current_step * env.time_interval
            
            # 1. 应用DR节点预选的战术
            if hasattr(self, 'next_round_tactic') and self.next_round_tactic:
                # 如果DR节点已经预选了战术，直接应用
                if self.selected_tactic != self.next_round_tactic:
                    old_tactic = self.selected_tactic
                    self.selected_tactic = self.next_round_tactic
                    logging.info(f"🔄 [MTR2决策] 应用DR预选战术: {old_tactic} → {self.selected_tactic}")
            else:
                # 否则重新选择战术
                logging.warning(f"⚠️ [MTR2决策] 未找到预选战术，重新选择")
                threat_info = self.threat_evaluator.evaluate_situation(env)
                threat_level = threat_info.get('situation', 'ADVANTAGE')
                enemy_intent_type = self._get_enemy_intent_type(env, 'A0100')
                
                # 查询决策表获取候选战术
                tactics, _ = self.decision_table.query_dr(self.my_intent, threat_level, enemy_intent_type)
                if tactics:
                    self.selected_tactic = tactics[0]
                    logging.info(f"🎯 [MTR2决策] 重新选择战术: {self.selected_tactic}")
            
            # 2. 威胁等级检查（保守策略下，敌方占优则终止）
            if self.my_intent == 'CONSERVATIVE_CLEAR':
                threat_info = self.threat_evaluator.evaluate_situation(env)
                situation = threat_info.get('situation', 'ADVANTAGE')
                if situation == 'DISADVANTAGE':
                    logging.warning(f"🚫 [MTR2决策] 保守策略下敌方占优，终止第二轮攻击")
                    self.should_reengage = False
                    self.is_second_attack = False
                    return
            
            # 3. 队形调整（根据新战术调整队形）
            logging.info(f"✅ [MTR2决策] 第二轮进攻准备完成，战术: {self.selected_tactic}")
            
        except Exception as e:
            logging.error(f"MTR2决策错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _decide_at_lr2(self, env, agent_id: str):
        """
        LR2节点决策：第二轮发射的威胁阈值检查
        核心任务：
        1. 检查威胁等级
        2. 如果威胁过高（>0.8），放弃导弹脱离
        3. 否则正常发射
        
        文档要求：LR1'/2'：发射, tl>0.8撤(弃导,脱离)
        """
        try:
            # 1. 检查发射次数
            if agent_id not in self.lr_launch_count:
                self.lr_launch_count[agent_id] = 0
            
            # 2. 威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            wing_threat = threat_info.get('wing_threat', 0.5)
            current_threat = lead_threat if agent_id == 'A0100' else wing_threat
            
            # 3. 威胁阈值检查
            if current_threat > 0.8:
                logging.warning(f"🚫 [LR2决策] {agent_id} 威胁等级{current_threat:.2f}>0.8，放弃导弹脱离")
                self.should_reengage = False
                self.is_second_attack = False
                # 不发射，直接返回
                return
            
            # 4. 威胁可接受，执行正常发射决策
            logging.info(f"✅ [LR2决策] {agent_id} 威胁等级{current_threat:.2f}可接受，准备第二次发射")
            self._decide_at_lr(env, agent_id)
            
        except Exception as e:
            logging.error(f"LR2决策错误 {agent_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())
    
    def _decide_at_tr2(self, env, agent_id: str):
        """
        TR2节点决策：第二轮规避评估
        核心任务：
        1. 第二次中制导结束
        2. 威胁再评估
        3. 准备脱离至MAR
        """
        try:
            # 威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            wing_threat = threat_info.get('wing_threat', 0.5)
            current_threat = lead_threat if agent_id == 'A0100' else wing_threat
            
            if current_threat > 0.7:
                logging.info(f"⚠️ [TR2决策] {agent_id} 威胁等级{current_threat:.2f}>0.7，准备紧急脱离")
            else:
                logging.info(f"✅ [TR2决策] {agent_id} 威胁等级{current_threat:.2f}，正常规避准备")
            
            # 执行标准TR决策
            self._decide_at_tr(env, agent_id)
            
        except Exception as e:
            logging.error(f"TR2决策错误 {agent_id}: {e}")
            import traceback
            logging.error(traceback.format_exc())

    def _compute_attack_approach_commands(self, env, agent_id: str):
        """计算占位接敌指令：对齐敌机方位、高度收敛、适度加速"""
        try:
            # 对齐敌机方位
            enemy_bearing = self._get_enemy_bearing(env, agent_id)
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            heading_diff = ((enemy_bearing - current_heading + 180) % 360) - 180
            hdg_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
            
            # 高度收敛（与敌机高度差减半）- 使用固定目标分配
            enemy_id = get_target_with_fallback(agent_id, env)
            if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                my_alt = env.agents[agent_id].get_position()[2]
                enemy_alt = env.agents[enemy_id].get_position()[2]
                alt_err = (enemy_alt - my_alt) * 0.5
            else:
                alt_err = 0.0
            alt_id = self._convert_altitude_to_index(alt_err)
            
            # 速度：距离越远越加速
            distance = self._calculate_distance(env)
            if distance > self.tactical_distances['MTR']:
                vel_delta = 100.0
            elif distance > self.tactical_distances['LR']:
                vel_delta = 50.0
            else:
                vel_delta = 0.0
            vel_id = self._convert_velocity_to_index(vel_delta)
            
            return alt_id, hdg_id, vel_id
        except Exception as _:
            return 7, 8, 3
    
    def _decide_at_mar(self, env, agent_id: str):
        """
        MAR节点决策：机动决策（强制脱离）
        核心任务：达到最小规避距离，强制脱离
        决策内容：执行返航机动
        """
        try:
            logging.warning(f"⚠️ [MAR决策] {agent_id} 达到最小规避距离(40km)，强制脱离")
            
            # 记录强制脱离状态
            if not hasattr(self, 'forced_retreat'):
                self.forced_retreat = {}
            self.forced_retreat[agent_id] = True
            
        except Exception as e:
            logging.error(f"MAR决策错误: {e}")
    
    def _normalize_angle_diff(self, angle_diff: float) -> float:
        """标准化角度差值到[-180, 180]"""
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        return angle_diff
    
    def _maintain_heading_precise(self, env, agent_id: str, target_heading: float, duration=10.0) -> tuple:
        """
        精确保持航向 - 严格按照原战术模板
        
        Args:
            env: 环境
            agent_id: 飞机ID
            target_heading: 目标航向（度）
            duration: 持续时间（秒）
        
        Returns:
            (altitude_cmd, heading_cmd, velocity_cmd)
        """
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180: heading_diff -= 360
        while heading_diff < -180: heading_diff += 360
        
        # 默认索引
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度
        
        # 精度控制：3度精度（防止过于灵敏）
        if abs(heading_diff) > 3.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def _get_wingman_phase_by_distance(self, distance: float) -> TacticalPhase:
        """
        僚机独立的战术阶段判断 - 体现时间线滞后
        从原战术模板移植，适配5阶段系统到8阶段系统
        
        原来的5阶段：NLT_MELD, MELD_MTR, MTR_TR, TR_DOR, DOR_DR
        现在的8阶段：NLT_MELD, MELD_MTR, MTR_LR, LR_TR, TR_DOR, DOR_DR, DR_MAR, BEYOND_MAR
        
        僚机使用滞后距离判断阶段（延迟 = 距离减少，更近才执行）
        """
        # 僚机滞后设置（从原战术模板）
        wingman_delay_tr_dor = 4000   # TR_DOR阶段滞后4km
        wingman_delay_dor_dr = 8000   # DOR_DR阶段滞后8km
        
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
    
    def _convert_heading_to_index(self, heading_diff_rad):
        """将航向差值（弧度）转换为索引"""
        idx = np.argmin(np.abs(self.norm_delta_heading - heading_diff_rad))
        return int(idx)
    
    def _convert_altitude_to_index(self, altitude_diff):
        """将高度差值（米）转换为索引"""
        altitude_diff_km = altitude_diff / 1000.0
        idx = np.argmin(np.abs(self.norm_delta_altitude - altitude_diff_km))
        return int(idx)
    
    def _convert_velocity_to_index(self, velocity_offset):
        """将速度差值（m/s）转换为索引"""
        velocity_offset_norm = velocity_offset / 100.0
        idx = np.argmin(np.abs(self.norm_delta_velocity - velocity_offset_norm))
        return int(idx)
    
    def _init_short_skate(self, agent_id, current_time, direction='auto'):
        """
        初始化short_skate机动状态 - 严格按照原战术模板
        
        Args:
            direction: 'auto'(根据队伍), 'left'(左转), 'right'(右转)
        """
        # 确定转向方向
        if direction == 'left':
            crank_angle = -40.0
            turn_cold_angle = -100.0
        elif direction == 'right':
            crank_angle = 40.0
            turn_cold_angle = 100.0
        else:  # 'auto' - 根据队伍
            crank_angle = -40.0 if agent_id.startswith('A') else 40.0
            turn_cold_angle = -100.0 if agent_id.startswith('A') else 100.0
        
        self.short_skate_states[agent_id] = {
            "phase": "crank",
            "phase_start_time": current_time,
            "total_start_time": current_time,
            "crank_angle": crank_angle,
            "turn_cold_angle": turn_cold_angle,
            "initial_heading": None,
            "initial_altitude": None
        }
        self.short_skate_start_time[agent_id] = current_time
    
    def _execute_short_skate_precise(self, env, agent_id, current_time, direction='auto'):
        """
        执行精确的Short Skate机动 - 严格按照原战术模板（简化版）
        
        Args:
            direction: 'auto'(根据队伍), 'left'(左转), 'right'(右转)
        """
        if agent_id not in self.short_skate_states:
            self._init_short_skate(agent_id, current_time, direction)
        
        state = self.short_skate_states[agent_id]
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        
        if state["initial_heading"] is None:
            state["initial_heading"] = current_heading
        
        phase_time = current_time - state["phase_start_time"]
        
        # 时间参数（僚机时间缩短，避免右转过久）
        if agent_id == "A0200":
            crank_duration = 12.0     # 缩短：18→12秒
            turn_cold_duration = 18.0  # 缩短：30→18秒
            escape_duration = 15.0     # 缩短：22→15秒
        else:
            crank_duration = 6.0
            turn_cold_duration = 15.0
            escape_duration = 15.0
        
        # 阶段1：Crank机动
        if state["phase"] == "crank":
            if phase_time < crank_duration:
                target_heading = state["initial_heading"] + state["crank_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "turn_cold"
                state["phase_start_time"] = current_time
                state["turn_cold_start_heading"] = current_heading
        
        # 阶段2：Turn Cold
        elif state["phase"] == "turn_cold":
            if phase_time < turn_cold_duration:
                target_heading = state["turn_cold_start_heading"] + state["turn_cold_angle"]
                target_heading = target_heading % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
            else:
                state["phase"] = "escape"
                state["phase_start_time"] = current_time
        
        # 阶段3：加速逃离
        elif state["phase"] == "escape":
            if phase_time < escape_duration:
                return 7, 8, 5  # 保持航向+加速
            else:
                # 完成，返航
                if agent_id.startswith('A'):
                    target_heading = 180.0
                else:
                    target_heading = 0.0
                
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                if abs(heading_diff) > 5.0:
                    return 7, 6 if heading_diff < 0 else 10, 3
                else:
                    return 7, 8, 3
        
        return 7, 8, 3
    
    def _convert_maneuver_result_to_indices(self, env, agent_id, target_heading, target_altitude,
                                           velocity_offset, target_roll, initial_heading, initial_altitude):
        """将pure_maneuvers的结果转换为索引 - 严格按照原战术模板"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_altitude = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        
        altitude_cmd_id = 7
        heading_cmd_id = 8
        velocity_cmd_id = 3
        
        # 高度控制
        if target_altitude is not None:
            altitude_diff = target_altitude - current_altitude
            if current_altitude < 2000.0 and altitude_diff < 0:
                altitude_diff = 200.0
            if abs(altitude_diff) > 5.0:
                altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        else:
            if initial_altitude is not None:
                altitude_diff = initial_altitude - current_altitude
                if current_altitude < 2000.0 and altitude_diff < 0:
                    altitude_diff = 200.0
                if abs(altitude_diff) > 10.0:
                    altitude_cmd_id = self._convert_altitude_to_index(altitude_diff)
        
        # 航向控制
        if target_heading is not None:
            heading_diff = target_heading - current_heading
            while heading_diff > 180: heading_diff -= 360
            while heading_diff < -180: heading_diff += 360
            
            if abs(heading_diff) > 1.0:
                heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        
        # 速度控制
        if velocity_offset is not None and abs(velocity_offset) > 2.0:
            velocity_cmd_id = self._convert_velocity_to_index(velocity_offset)
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def _record_simulation_data(self, env, current_time: float):
        """
        记录仿真数据
        记录轨迹、雷达、导弹、威胁、决策、阶段等所有数据
        """
        try:
            # 1. 记录轨迹数据
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    pos = env.agents[agent_id].get_position()
                    vel = env.agents[agent_id].get_velocity()
                    heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                    pitch = env.agents[agent_id].get_property_value(c.attitude_theta_deg)
                    roll = env.agents[agent_id].get_property_value(c.attitude_phi_deg)
                    
                    self.data_logger.log_trajectory(
                        current_time, agent_id, pos, vel, heading, pitch, roll
                    )
            
            # 2. 记录雷达状态（模拟）
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    # 获取敌机作为目标（使用固定目标分配）
                    target_id = get_target_with_fallback(agent_id, env)
                    
                    # 计算距离
                    if target_id in env.agents and env.agents[target_id].is_alive:
                        my_pos = env.agents[agent_id].get_position()
                        target_pos = env.agents[target_id].get_position()
                        distance = np.linalg.norm(target_pos - my_pos)
                        
                        # 模拟雷达模式
                        if distance > 100000:
                            mode = 'SEARCH'
                            lock_quality = 0.0
                        elif distance > 80000:
                            mode = 'TRACK'
                            lock_quality = 0.6
                        else:
                            mode = 'LOCK'
                            lock_quality = 0.9
                        
                        snr = 20.0 if mode == 'LOCK' else 15.0 if mode == 'TRACK' else 10.0
                        
                        self.data_logger.log_radar_status(
                            current_time, agent_id, mode, target_id, distance, lock_quality, snr
                        )
            
            # 3. 记录导弹状态
            if hasattr(self, 'missiles'):
                for missile_id, missile in self.missiles.items():
                    if missile.is_alive:
                        pos = missile.get_position()
                        speed = np.linalg.norm(missile.get_velocity())
                        target_id = missile.target_aircraft.uid if missile.target_aircraft else 'UNKNOWN'
                        
                        # 计算到目标距离
                        if missile.target_aircraft and missile.target_aircraft.is_alive:
                            distance_to_target = missile.target_distance
                        else:
                            distance_to_target = 0
                        
                        # 制导模式
                        guidance_mode = 'TERMINAL' if missile._phase == 2 else 'MID_COURSE'
                        
                        # 获取发射者ID
                        launcher_id = missile.parent_aircraft.uid if missile.parent_aircraft else 'UNKNOWN'
                        
                        self.data_logger.log_missile_status(
                            current_time, missile_id, launcher_id, target_id,
                            pos, speed, distance_to_target, guidance_mode
                        )
            
            # 4. 记录威胁评估（仅我方）
            if hasattr(self, 'threat_evaluator'):
                try:
                    threat_info = self.threat_evaluator.evaluate_situation(env)
                    
                    # 长机威胁
                    if 'A0100' in env.agents and env.agents['A0100'].is_alive:
                        self.data_logger.log_threat_assessment(
                            current_time, 'A0100',
                            threat_info.get('lead_threat', 0.5),
                            threat_info.get('distance_threat', 0.0),
                            threat_info.get('angle_threat', 0.0),
                            threat_info.get('altitude_threat', 0.0),
                            threat_info.get('speed_threat', 0.0),
                            threat_info.get('lead_threat', 0.5)
                        )
                    
                    # 僚机威胁
                    if 'A0200' in env.agents and env.agents['A0200'].is_alive:
                        self.data_logger.log_threat_assessment(
                            current_time, 'A0200',
                            threat_info.get('wingman_threat', 0.5),
                            threat_info.get('distance_threat', 0.0),
                            threat_info.get('angle_threat', 0.0),
                            threat_info.get('altitude_threat', 0.0),
                            threat_info.get('speed_threat', 0.0),
                            threat_info.get('wingman_threat', 0.5)
                        )
                except Exception as e:
                    pass  # 威胁评估失败不影响仿真
            
            # 5. 记录战术阶段（仅我方）
            distance = self._calculate_distance(env)
            for agent_id in ['A0100', 'A0200']:
                if agent_id in env.agents and env.agents[agent_id].is_alive:
                    phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
                    my_intent = getattr(self, 'my_intent', 'CONSERVATIVE_CLEAR')
                    
                    # 获取敌机意图（模拟）
                    enemy_intent = 'NEUTRAL'
                    
                    # 态势判断
                    if hasattr(self, 'threat_evaluator'):
                        try:
                            threat_info = self.threat_evaluator.evaluate_situation(env)
                            threat = threat_info.get('lead_threat' if agent_id == 'A0100' else 'wingman_threat', 0.5)
                            situation = 'ADVANTAGE' if threat < 0.4 else 'NEUTRAL' if threat < 0.7 else 'DISADVANTAGE'
                        except:
                            situation = 'NEUTRAL'
                    
                    self.data_logger.log_tactical_phase(
                        current_time, agent_id, phase.value, distance, my_intent, enemy_intent, situation
                    )
            
        except Exception as e:
            logging.warning(f"记录数据失败: {e}")
            # 记录失败不影响仿真继续
    
    def _check_cap_boundary(self, env, agent_id: str) -> tuple:
        """
        【完整修复】检查是否超出CAP边界（跑马道形状）
        修复：
        1. 正确判断跑马道形状（矩形+两端半圆）
        2. 增强边界返回逻辑，强制返回而非柔和调整
        3. 限制横向分离距离，确保不超出CAP边界

        Returns:
            (is_out_of_bounds, return_command)
            - is_out_of_bounds: bool, 是否严重越界
            - return_command: tuple or None, 如果严重越界则返回强制调整指令
        """
        try:
            pos = env.agents[agent_id].get_position()
            x, y, z = pos[0], pos[1], pos[2]

            # 检查高度
            out_z = z < self.cap_boundary['z_min'] or z > self.cap_boundary['z_max']

            # 检查是否在跑马道内（南北向椭圆）
            rect_x_min = self.cap_boundary['rect_x_min']
            rect_x_max = self.cap_boundary['rect_x_max']
            rect_y_min = self.cap_boundary['rect_y_min']
            rect_y_max = self.cap_boundary['rect_y_max']
            radius = self.cap_boundary['circle_radius']
            south_center = self.cap_boundary['south_circle_center']
            north_center = self.cap_boundary['north_circle_center']

            in_boundary = False
            in_warning_zone = False  # 接近边界警告区

            # 【修复1】正确判断跑马道形状
            # 检查是否在矩形部分内（中间直道）
            if rect_x_min <= x <= rect_x_max:
                # 在矩形X范围内，检查Y是否在范围内
                if rect_y_min <= y <= rect_y_max:
                    in_boundary = True
                    # 检查是否接近Y边界（警告区：距离边界5km以内）
                    if abs(y - rect_y_max) < 5000 or abs(y - rect_y_min) < 5000:
                        in_warning_zone = True
                else:
                    # Y超出矩形范围，但可能在两端半圆内
                    # 检查是否在南端半圆内（X接近南端）
                    if x < 0:  # 靠近南端
                        dist_to_south = np.sqrt((x - south_center[0])**2 + (y - south_center[1])**2)
                        if dist_to_south <= radius:
                            in_boundary = True
                            if dist_to_south > radius * 0.9:  # 距离半圆边界10%以内
                                in_warning_zone = True
                    # 检查是否在北端半圆内（X接近北端）
                    else:  # 靠近北端
                        dist_to_north = np.sqrt((x - north_center[0])**2 + (y - north_center[1])**2)
                        if dist_to_north <= radius:
                            in_boundary = True
                            if dist_to_north > radius * 0.9:  # 距离半圆边界10%以内
                                in_warning_zone = True
            # 检查是否在南端半圆内（X < 矩形南边界）
            elif x < rect_x_min:
                dist_to_south = np.sqrt((x - south_center[0])**2 + (y - south_center[1])**2)
                if dist_to_south <= radius:
                    in_boundary = True
                    if dist_to_south > radius * 0.9:  # 距离半圆边界10%以内
                        in_warning_zone = True
            # 检查是否在北端半圆内（X > 矩形北边界）
            elif x > rect_x_max:
                dist_to_north = np.sqrt((x - north_center[0])**2 + (y - north_center[1])**2)
                if dist_to_north <= radius:
                    in_boundary = True
                    if dist_to_north > radius * 0.9:  # 距离半圆边界10%以内
                        in_warning_zone = True

            # 如果在实际边界内且高度正常，返回False - 允许战术自由执行
            if in_boundary and not out_z:
                # 如果在警告区，输出警告但不强制返回
                if in_warning_zone and env.current_step % 200 == 0:
                    logging.info(f"⚠️ [{agent_id}] 接近CAP边界! "
                               f"位置:({x/1000:.1f},{y/1000:.1f},{z/1000:.1f}km)")
                return False, None

            # 【修复2】超出边界时，强制返回中心
            # 计算返回目标点
            # 优先返回到最近的安全区域
            if abs(y) > rect_y_max:
                # Y方向超出，返回到Y=0中心线
                target_x = np.clip(x, rect_x_min + 5000, rect_x_max - 5000)  # 保持在矩形X范围内，留5km缓冲
                target_y = 0  # 返回中心线
            elif x < rect_x_min:
                # X方向超出南端，返回到南端矩形边界
                target_x = rect_x_min + 5000  # 进入矩形区域5km
                target_y = np.clip(y, -30000, 30000)  # 限制Y在±30km范围内
            elif x > rect_x_max:
                # X方向超出北端，返回到北端矩形边界
                target_x = rect_x_max - 5000  # 进入矩形区域5km
                target_y = np.clip(y, -30000, 30000)  # 限制Y在±30km范围内
            else:
                # 其他情况，返回到中心点
                target_x = 0
                target_y = 0

            # 计算返回航向
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            delta_x = target_x - x
            delta_y = target_y - y

            # 计算目标方位角（从北向顺时针）
            target_bearing = np.rad2deg(np.arctan2(delta_y, delta_x))
            target_heading = (90 - target_bearing) % 360.0

            heading_diff = ((target_heading - current_heading + 180) % 360) - 180

            # 【修复3】强制返回：使用更大的航向调整
            if abs(heading_diff) > 90:
                heading_cmd = 16 if heading_diff > 0 else 0  # ±180度大转弯
            elif abs(heading_diff) > 60:
                heading_cmd = 15 if heading_diff > 0 else 1  # ±120度
            elif abs(heading_diff) > 30:
                heading_cmd = 13 if heading_diff > 0 else 3  # ±75度
            elif abs(heading_diff) > 15:
                heading_cmd = 11 if heading_diff > 0 else 5  # ±45度
            else:
                heading_cmd = 9 if heading_diff > 0 else 7  # ±15度

            # 选择高度指令
            if z < self.cap_boundary['z_min']:
                alt_cmd = 13  # 爬升+1000m（更强力）
            elif z > self.cap_boundary['z_max']:
                alt_cmd = 1   # 下降-1000m（更强力）
            else:
                alt_cmd = 7   # 保持高度

            # 【修复4】降低日志频率，避免刷屏
            if env.current_step % 120 == 0:  # 每120步（24秒）输出一次
                logging.warning(f"🚨 [{agent_id}] 超出CAP边界! "
                            f"位置:({x/1000:.1f},{y/1000:.1f},{z/1000:.1f}km) "
                            f"目标:({target_x/1000:.1f},{target_y/1000:.1f}km) "
                            f"航向调整{heading_diff:+.1f}°")
            return True, (alt_cmd, heading_cmd, 3)  # 强制返回（保持速度）

        except Exception as e:
            logging.error(f"CAP边界检查失败: {e}")
            return False, None
    
    def _get_tactical_command_indices(self, env, agent_id: str) -> tuple:
        """
        根据战术决策返回指令索引 - 完整战术分发系统
        返回: (altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
        """
        # ===== 优先级0: 我方CAP边界约束（最高优先级） =====
        if agent_id.startswith('A'):
            is_out, return_cmd = self._check_cap_boundary(env, agent_id)
            if is_out:
                return return_cmd
        
        # ===== 敌方AI系统：使用统一敌方战术AI =====
        if agent_id.startswith('B'):
            # ✅ 启用完整敌方AI系统
            if hasattr(self, 'enemy_ai'):
                current_time = env.current_step * env.time_interval
                return self.enemy_ai.get_enemy_command_indices(env, agent_id, current_time)
            else:
                # 降级：如果AI系统未初始化，使用简化逻辑
                logging.warning(f"⚠️ 敌方AI系统未初始化，使用默认平飞")
                return 7, 8, 3
        
        # ===== 旧版：简化直飞+返航逻辑（已弃用） =====
        # if agent_id.startswith('B'):
        #     # 初始化敌方返航状态
        #     if not hasattr(self, 'enemy_return_state'):
        #         self.enemy_return_state = {}
        #     
        #     if agent_id not in self.enemy_return_state:
        #         self.enemy_return_state[agent_id] = {
        #             'returning': False,
        #             'logged': False
        #         }
        #     
        #     # 获取距离我方长机的距离（2D水平距离）
        #     enemy_pos = np.array(env.agents[agent_id].get_position())
        #     if 'A0100' not in env.agents or not env.agents['A0100'].is_alive:
        #         # 长机不存在，继续前进
        #         return 7, 8, 3
        #     lead_pos = np.array(env.agents['A0100'].get_position())
        #     # 只计算水平距离（x, y），忽略高度差
        #     distance = np.linalg.norm(enemy_pos[:2] - lead_pos[:2])
        #     
        #     # ✅ 修复：返航距离改为80km（避免最后才触发），并添加调试日志
        #     # 每60步（12秒）打印一次距离信息
        #     if env.current_step % 60 == 0 and agent_id == 'B0200':
        #         current_time = env.current_step * env.time_interval
        #         logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 当前距离我方长机{distance/1000:.1f}km")
        #     
        #     # 距离小于80km时，执行180度转弯返航
        #     if distance < 65000 and not self.enemy_return_state[agent_id]['returning']:
        #         self.enemy_return_state[agent_id]['returning'] = True
        #         if not self.enemy_return_state[agent_id]['logged']:
        #             current_time = env.current_step * env.time_interval
        #             logging.info(f"[T={current_time:.1f}s][敌方{agent_id}] 距离{distance/1000:.1f}km < 65km，🔄开始返航")
        #             self.enemy_return_state[agent_id]['logged'] = True
        #     
        #     # 返航：航向0度（朝北），保持高度，中速
        #     if self.enemy_return_state[agent_id]['returning']:
        #         current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        #         heading_diff = (0 - current_heading + 180) % 360 - 180
        #         
        #         # 已经接近0度航向，保持直飞
        #         if abs(heading_diff) < 10:
        #             return 7, 8, 3  # 保持高度，直飞，中速
        #         # 需要转向到0度
        #         elif heading_diff > 0:
        #             return 7, 10, 3  # 保持高度，右转30度，中速
        #         else:
        #             return 7, 6, 3   # 保持高度，左转30度，中速
        #     
        #     # 正常前进：保持当前航向，中速
        #     return 7, 8, 3
        
        # ===== 我方战术系统 =====
        # 只对我方（A开头）执行战术系统
        if not agent_id.startswith('A'):
            logging.error(f"⚠️ 错误：敌方{agent_id}进入我方战术系统！")
            return 7, 8, 3  # 平飞
        
        # 1. 更新全局阶段
        self._update_tactical_phase(env)
        
        # 2. 更新该飞机的独立阶段
        self._update_tactical_phase(env, agent_id)
        
        # 3. 获取选定战术（优先使用决策系统的结果）
        if self.selected_tactic is None:
            # 如果还没有决策，使用默认战术
            self.selected_tactic = 'PINCER_ATTACK'
        
        selected_tactic = self.selected_tactic
        
        # ✅ 修复：战术模板优先于formation_commands
        # 原因：formation_commands会覆盖钳形攻势等战术模板的精确指令
        agent_phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
        
        # 3.6 DOR/DR/MAR阶段：优先按规避/再进攻/返航统一路径执行
        if agent_phase in [TacticalPhase.DOR_DR, TacticalPhase.DR_MAR, TacticalPhase.BEYOND_MAR]:
            result = self._execute_tactical_evasion(env, agent_id)
        # 4. 根据战术类型分发（7种完整战术）- 优先于formation_commands
        elif selected_tactic == 'DRAG_SHOOT':
            result = self._execute_drag_shoot(env, agent_id)
        elif selected_tactic == 'PINCER_ATTACK':
            result = self._execute_pincer_attack(env, agent_id)
        elif selected_tactic == 'HIGH_LOW_ATTACK':
            result = self._execute_high_low_attack(env, agent_id)
        elif selected_tactic == 'FRONT_BACK':
            result = self._execute_front_back(env, agent_id)
        elif selected_tactic == 'TACTICAL_EVASION':
            result = self._execute_tactical_evasion(env, agent_id)
        elif selected_tactic == 'TACTICAL_TURN':
            result = self._execute_tactical_turn(env, agent_id)
        else:  # SIDE_BY_SIDE 或默认
            result = self._execute_side_by_side(env, agent_id)
        
        # ===== 全局安全检查：高度和速度限制 =====
        current_alt = env.agents[agent_id].get_position()[2]
        current_speed = env.agents[agent_id].get_property_value(c.velocities_v_north_mps)**2 + \
                       env.agents[agent_id].get_property_value(c.velocities_v_east_mps)**2 + \
                       env.agents[agent_id].get_property_value(c.velocities_v_down_mps)**2
        current_speed = np.sqrt(current_speed)

        alt_cmd, hdg_cmd, vel_cmd = result

        # 初始化高度保护状态追踪（避免日志泛滥）
        if not hasattr(self, 'altitude_protection_state'):
            self.altitude_protection_state = {}
        
        if agent_id not in self.altitude_protection_state:
            self.altitude_protection_state[agent_id] = {
                'last_warning_type': None,  # ✅ 修复：只记录类型，不记录完整字符串
                'last_log_time': 0
            }
        
        # 高度安全检查：防止超过10km上限或低于3km下限
        alt_change = self.norm_delta_altitude[alt_cmd] * 1000
        predicted_alt = current_alt + alt_change
        current_time = env.current_step * env.time_interval
        
        # 判断当前警告类型和消息
        warning_type = None  # ✅ 修复：使用类型而不是完整字符串
        warning_message = None
        
        # 【修复7+终极增强】更严格的高度保护：7.5km开始强制下降
        # 严格的高度限制：7.5km强制下降，8.5km强制大幅下降，9km强制最大下降
        if current_alt > 9000:
            # 已经超过9km，强制最大下降
            alt_cmd = 0  # 下降-1500m（最大下降）
            vel_cmd = np.clip(vel_cmd, 0, 3)  # 限制加速，避免爬升
            warning_type = 'over_9km'
            warning_message = f"🛡️ [{agent_id}] 高度保护: 当前{current_alt:.0f}m > 9km上限，强制最大下降"
        elif current_alt > 8500:
            # 超过8.5km，强制大幅下降
            alt_cmd = 1  # 下降-1000m
            vel_cmd = np.clip(vel_cmd, 0, 3)  # 限制加速
            warning_type = 'over_8.5km'
            warning_message = f"🛡️ [{agent_id}] 高度保护: 当前{current_alt:.0f}m > 8.5km，强制大幅下降"
        elif current_alt > 7500:
            # 超过7.5km，强制下降（不再只是禁止爬升）
            if alt_change > 0:
                alt_cmd = 3  # 改为下降-500m（更激进）
                warning_type = 'over_7.5km_force_down'
                warning_message = f"🛡️ [{agent_id}] 高度保护: 当前{current_alt:.0f}m > 7.5km，强制下降"
            else:
                # 如果已经在下降，继续下降
                pass
        elif predicted_alt > 10000:
            # 预测会超过10km，强制下降
            alt_cmd = 3  # 下降-500m
            warning_type = 'predict_over_10km'
            warning_message = f"🛡️ [{agent_id}] 高度保护: 预测{predicted_alt:.0f}m > 10km上限，强制下降"
        
        # ✅ 修复：只有当警告类型改变或距离上次日志超过10秒时才输出
        if warning_type:
            state = self.altitude_protection_state[agent_id]
            should_log = False
            
            if state['last_warning_type'] != warning_type:
                # 警告类型改变，立即输出日志
                should_log = True
                state['last_warning_type'] = warning_type
                state['last_log_time'] = current_time
            elif current_time - state['last_log_time'] > 10.0:  # ✅ 从5秒改为10秒
                # 同一类型超过10秒，输出一次
                should_log = True
                state['last_log_time'] = current_time
            
            if should_log:
                logging.warning(warning_message)

        # 下限检查
        if current_alt < 3000:
            # 已经低于3km，强制爬升
            alt_cmd = 14  # 爬升+1500m（最大爬升）
            logging.warning(f"🛡️ [{agent_id}] 高度保护: 当前{current_alt:.0f}m < 3km下限，强制最大爬升")
        elif current_alt < 3500:
            # 接近3km下限，禁止下降
            if alt_change < 0:
                alt_cmd = 7  # 改为保持高度
                logging.warning(f"🛡️ [{agent_id}] 高度保护: 当前{current_alt:.0f}m接近3km下限，禁止下降")
        elif predicted_alt < 3000:
            # 预测会低于3km，改为保持高度
            alt_cmd = 7  # 保持高度
            logging.warning(f"🛡️ [{agent_id}] 高度保护: 预测{predicted_alt:.0f}m < 3km下限，改为保持高度")

        # 【修复8+增强】DR_MAR阶段更严格的高度和机动限制
        agent_phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
        if agent_phase in [TacticalPhase.DOR_DR, TacticalPhase.DR_MAR, TacticalPhase.BEYOND_MAR]:
            # DR_MAR阶段：严格限制高空机动
            if current_alt > 8000:
                # 高空时，禁止大幅加速
                vel_cmd = np.clip(vel_cmd, 0, 4)  # 最多+50m/s
                if env.current_step % 200 == 0:
                    logging.info(f"🛡️ [{agent_id}] DR_MAR高空限速: 高度{current_alt:.0f}m，限制加速")
            
            # ✅ 新增：7.5km以上禁止大角度转向（避免爬升）
            if current_alt > 7500:
                # 限制转向角度，避免大角度转向导致爬升
                if hdg_cmd in [2, 14]:  # 90度转向
                    hdg_cmd = 5 if hdg_cmd == 2 else 11  # 改为45度
                elif hdg_cmd in [4, 12]:  # 60度转向
                    hdg_cmd = 5 if hdg_cmd == 4 else 11  # 改为45度
                if env.current_step % 200 == 0:
                    logging.info(f"🛡️ [{agent_id}] DR_MAR高空限转: 高度{current_alt:.0f}m，限制大角度转向")

        # 速度安全检查：防止高空低速失速
        if current_alt > 9000 and current_speed < 150:
            # 极高空低速，强制下降换速度
            alt_cmd = 1  # 下降-1000m（加大下降力度）
            vel_cmd = 5  # 大幅加速+100m/s
            logging.warning(f"⚠️ [{agent_id}] 极高空低速! 强制下降换速度 (高度{current_alt:.0f}m, 速度{current_speed:.1f}m/s)")
        elif current_alt > 8000 and current_speed < 120:
            # 高空极低速，强制下降换速度
            alt_cmd = 3  # 下降-500m
            vel_cmd = 4  # 加速+50m/s
            logging.warning(f"⚠️ [{agent_id}] 高空极低速! 强制下降换速度 (高度{current_alt:.0f}m, 速度{current_speed:.1f}m/s)")
        
        # ✅ 关键修复：完全禁用Y轴约束，因为会导致90度转向和转圈
        # 原因：Y轴约束触发后，_turn_to_heading会执行90度转向，导致持续转圈
        # 解决方案：依赖CAP边界检查，不在此处强制转向
        # if agent_id.startswith('A'):  # 仅对我方飞机约束
        #     my_pos = env.agents[agent_id].get_position()
        #     y_pos = my_pos[1]  # Y坐标（东西方向）
        #     x_pos = my_pos[0]  # X坐标（南北方向）
        #     
        #     # ✅ 关键修复2：Y轴约束从40km降至25km
        #     if abs(y_pos) > 25000:
        #         # 计算目标航向：返回Y=0中心线，同时继续X轴战斗
        #         current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        #         
        #         # 根据X位置决定主航向（南北）
        #         if x_pos < 0:  # 我方在南方
        #             main_heading = 0  # 向北进攻
        #         else:
        #             main_heading = 180  # 向南返回
        #         
        #         # 根据Y偏离方向微调航向，使其向中心线靠拢
        #         if y_pos > 0:  # 偏东，需要向西修正
        #             target_heading = (main_heading - 30) % 360  # 偏西30度
        #         else:  # 偏西，需要向东修正
        #             target_heading = (main_heading + 30) % 360  # 偏东30度
        #         
        #         # 计算航向差并应用
        #         heading_diff = self._normalize_angle_diff(target_heading - current_heading)
        #         hdg_cmd = self._convert_heading_to_index(np.deg2rad(heading_diff))
        #         
        #         if env.current_step % 60 == 0:
        #             logging.warning(f"⚠️ [{agent_id}] Y轴约束: Y={y_pos/1000:.1f}km>25km, 强制转向{target_heading:.0f}° (返回中心线)")

        result = (alt_cmd, hdg_cmd, vel_cmd)

        # 调试日志：打印指令和机动类型
        if env.current_step % 30 == 0:
            alt_change = self.norm_delta_altitude[result[0]] * 1000
            hdg_change = np.rad2deg(self.norm_delta_heading[result[1]])
            vel_change = self.norm_delta_velocity[result[2]] * 100

            # 判断机动类型
            maneuver_type = "平飞"
            if abs(hdg_change) > 80:
                maneuver_type = "180°返航" if abs(hdg_change) > 170 else "90°转向"
            elif abs(hdg_change) > 40:
                maneuver_type = f"Crank{hdg_change:+.0f}°"
            elif abs(hdg_change) > 10:
                maneuver_type = f"转向{hdg_change:+.0f}°"

            if abs(alt_change) > 200:
                maneuver_type += f" + {'爬升' if alt_change > 0 else '俯冲'}{abs(alt_change):.0f}m"

            # 机动指令日志（已注释，减少输出）
            # logging.warning(f"[机动指令] {agent_id}")
            # logging.warning(f"  机动类型: {maneuver_type}")
            # logging.warning(f"  高度变化: {alt_change:+.0f}m | 航向变化: {hdg_change:+.1f}° | 速度变化: {vel_change:+.0f}m/s")
            # logging.warning(f"  指令索引: alt={result[0]} hdg={result[1]} vel={result[2]}")

        return result
    
    def _execute_notch_back(self, env, agent_id: str, direction='left') -> tuple:
        """
        执行Notch Back机动 - 新定义
        飞机做180°角的回旋机动的同时下降高度，以高度换速度更快远离敌机
        
        Args:
            direction: 'left' 左侧，'right' 右侧
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 初始化Notch Back状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'notch_back':
            self.maneuver_states[agent_id] = {
                'type': 'notch_back',
                'phase': 'turn_descend',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading
            }
            logging.info(f"🔄 【Notch Back】{agent_id}开始180°回旋下降机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        # 180°回旋+下降（持续8秒）
        if state['phase'] == 'turn_descend':
            if phase_time < 8.0:
                # 180°转向 + 下降-500m + 大幅加速（以高度换速度）
                return 3, 16, 5  # alt_cmd=3(-500m), hdg_cmd=16(180°), vel_cmd=5(+100m/s)
            else:
                del self.maneuver_states[agent_id]
                logging.info(f"✅ 【Notch Back】{agent_id}完成机动")
                return 7, 8, 4  # 恢复平飞
        
        return 7, 8, 4  # 默认平飞
    
    def _execute_tactical_crank(self, env, agent_id: str, direction='left', climb=True) -> tuple:
        """
        执行战术Crank机动 - 新定义
        飞机做crank机动的同时做爬升或下降机动，最后恢复至平飞状态，结束时方位角发生变化
        
        Args:
            direction: 'left' 左转，'right' 右转
            climb: True爬升，False下降
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 初始化状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'tactical_crank':
            self.maneuver_states[agent_id] = {
                'type': 'tactical_crank',
                'phase': 'crank_climb',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'climb': climb,
                'initial_heading': current_heading
            }
            action = "爬升" if climb else "下降"
            logging.info(f"🔄 【战术Crank】{agent_id}开始{direction}侧Crank+{action}机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        direction = state.get('direction', 'left')
        climb = state.get('climb', True)
        
        # 阶段1: Crank+爬升/下降（持续5秒）
        if state['phase'] == 'crank_climb':
            if phase_time < 5.0:
                alt_cmd = 11 if climb else 3  # 爬升+500m 或 下降-500m
                hdg_cmd = 4 if direction == 'left' else 12  # 左转60° 或 右转60°
                return alt_cmd, hdg_cmd, 4  # 加速
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术Crank】{agent_id}: Crank→平飞恢复")
        
        # 阶段2: 恢复平飞（持续2秒）
        if state['phase'] == 'level_off':
            if phase_time < 2.0:
                return 7, 8, 3  # 保持平飞
            else:
                del self.maneuver_states[agent_id]
                logging.info(f"✅ 【战术Crank】{agent_id}完成机动，方位角已变化")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def _execute_tactical_climb(self, env, agent_id: str, direction='left') -> tuple:
        """
        执行战术爬升机动 - 新定义
        飞机做crank机动的同时爬升，在爬升中途做相反方向的crank机动，
        最后回复平飞状态，机动结束时方位角不变
        
        Args:
            direction: 'left' 先左转，'right' 先右转
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 初始化状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'tactical_climb':
            self.maneuver_states[agent_id] = {
                'type': 'tactical_climb',
                'phase': 'crank1_climb',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading
            }
            logging.info(f"🔄 【战术爬升】{agent_id}开始{direction}侧战术爬升机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        direction = state.get('direction', 'left')
        
        # 阶段1: 第一次Crank+爬升（持续3秒）
        if state['phase'] == 'crank1_climb':
            if phase_time < 3.0:
                hdg_cmd = 4 if direction == 'left' else 12  # 左转60° 或 右转60°
                return 11, hdg_cmd, 4  # 爬升+500m + Crank + 加速
            else:
                state['phase'] = 'crank2_climb'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术爬升】{agent_id}: Crank1→Crank2")
        
        # 阶段2: 反向Crank+继续爬升（持续3秒）
        if state['phase'] == 'crank2_climb':
            if phase_time < 3.0:
                hdg_cmd = 12 if direction == 'left' else 4  # 反向Crank
                return 11, hdg_cmd, 4  # 继续爬升 + 反向Crank + 加速
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术爬升】{agent_id}: Crank2→平飞")
        
        # 阶段3: 恢复平飞（持续2秒）
        if state['phase'] == 'level_off':
            if phase_time < 2.0:
                return 7, 8, 3  # 保持平飞
            else:
                del self.maneuver_states[agent_id]
                logging.info(f"✅ 【战术爬升】{agent_id}完成机动，方位角不变")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def _execute_tactical_descend(self, env, agent_id: str, direction='left') -> tuple:
        """
        执行战术下降机动 - 新定义
        飞机做crank机动的同时下降，在下降中途做相反方向的crank机动，
        最后回复平飞状态，机动结束时方位角不变
        
        Args:
            direction: 'left' 先左转，'right' 先右转
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 初始化状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'tactical_descend':
            self.maneuver_states[agent_id] = {
                'type': 'tactical_descend',
                'phase': 'crank1_descend',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading
            }
            logging.info(f"🔄 【战术下降】{agent_id}开始{direction}侧战术下降机动")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        direction = state.get('direction', 'left')
        
        # 阶段1: 第一次Crank+下降（持续3秒）
        if state['phase'] == 'crank1_descend':
            if phase_time < 3.0:
                hdg_cmd = 4 if direction == 'left' else 12  # 左转60° 或 右转60°
                return 3, hdg_cmd, 4  # 下降-500m + Crank + 加速
            else:
                state['phase'] = 'crank2_descend'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术下降】{agent_id}: Crank1→Crank2")
        
        # 阶段2: 反向Crank+继续下降（持续3秒）
        if state['phase'] == 'crank2_descend':
            if phase_time < 3.0:
                hdg_cmd = 12 if direction == 'left' else 4  # 反向Crank
                return 3, hdg_cmd, 4  # 继续下降 + 反向Crank + 加速
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【战术下降】{agent_id}: Crank2→平飞")
        
        # 阶段3: 恢复平飞（持续2秒）
        if state['phase'] == 'level_off':
            if phase_time < 2.0:
                return 7, 8, 3  # 保持平飞
            else:
                del self.maneuver_states[agent_id]
                logging.info(f"✅ 【战术下降】{agent_id}完成机动，方位角不变")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def _execute_short_skate(self, env, agent_id: str, direction='left', target_heading=None) -> tuple:
        """
        执行Short Skate机动 - 参考拖曳射击实现
        三阶段：小角度Crank → 快速转向目标航向 → 加速逃离
        
        参考拖曳射击的实现：
        - 阶段1: Crank 40度（6秒）
        - 阶段2: Turn Cold 快速转向（15秒）
        - 阶段3: 加速逃离（15秒）
        
        Args:
            direction: 'left' 左侧返航（长机），'right' 右侧返航（僚机）
            target_heading: 目标返航航向（度），如果为None则自动判断
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_alt = env.agents[agent_id].get_position()[2]
        
        # 初始化Short Skate状态（关键：确保只初始化一次）
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'short_skate':
            # 确定目标航向
            if target_heading is None:
                if agent_id.startswith('A'):  # 我方：返回180°（南向）
                    target_heading = 180.0
                elif agent_id.startswith('B'):  # 敌方：返回0°（北向）
                    target_heading = 0.0
                else:
                    target_heading = 180.0
            
            # 记录初始航向
            initial_heading = current_heading
            
            self.maneuver_states[agent_id] = {
                'type': 'short_skate',
                'phase': 'crank',
                'start_time': current_time,
                'phase_start_time': current_time,
                'direction': direction,
                'initial_heading': initial_heading,
                'target_heading': target_heading,
                'crank_angle': -40.0 if direction == 'left' else 40.0,  # 左转-40°，右转+40°
                'turn_cold_angle': -100.0 if direction == 'left' else 100.0  # 继续转100°到达返航航向
            }
            logging.info(f"🔄 【Short Skate】{agent_id}开始{direction}侧返航机动 (初始{initial_heading:.1f}°→目标{target_heading:.1f}°)")
        
        state = self.maneuver_states[agent_id]
        phase_time = current_time - state['phase_start_time']
        
        # 阶段1: Crank机动（6秒）- 小角度偏置
        if state['phase'] == 'crank':
            crank_duration = 6.0
            if phase_time < crank_duration:
                # 计算目标航向：初始航向 + crank角度
                target_heading_now = state['initial_heading'] + state['crank_angle']
                target_heading_now = target_heading_now % 360
                heading_diff = self._normalize_angle_diff(target_heading_now - current_heading)
                
                if env.current_step % 30 == 0:
                    logging.info(f"🔄 [Short Skate-Crank] {agent_id} 当前{current_heading:.1f}°→目标{target_heading_now:.1f}° 差{heading_diff:.1f}° 高度{current_alt:.0f}m")
                
                if abs(heading_diff) > 5.0:
                    if heading_diff < 0:
                        return 7, 6, 3  # 左转30°
                    else:
                        return 7, 10, 3  # 右转30°
                else:
                    return 7, 8, 3  # 保持航向
            else:
                # 进入Turn Cold阶段
                state['phase'] = 'turn_cold'
                state['phase_start_time'] = current_time
                state['turn_cold_start_heading'] = current_heading
                logging.info(f"🔄 【Short Skate】{agent_id}: Crank→Turn Cold (当前{current_heading:.1f}°)")
        
        # 阶段2: Turn Cold - 快速转向目标航向（15秒）
        if state['phase'] == 'turn_cold':
            turn_cold_duration = 15.0
            if phase_time < turn_cold_duration:
                # 直接转向最终目标航向
                target_heading_now = state['target_heading']
                heading_diff = self._normalize_angle_diff(target_heading_now - current_heading)
                
                if env.current_step % 30 == 0:
                    logging.info(f"🔄 [Short Skate-TurnCold] {agent_id} 当前{current_heading:.1f}°→目标{target_heading_now:.1f}° 差{heading_diff:.1f}° 高度{current_alt:.0f}m")
                
                if abs(heading_diff) > 5.0:
                    # 使用更大的转弯指令快速转向
                    if heading_diff < 0:
                        return 7, 5, 3  # 左转45°（快速）
                    else:
                        return 7, 11, 3  # 右转45°（快速）
                else:
                    return 7, 8, 3  # 保持航向
            else:
                # 进入加速逃离阶段
                state['phase'] = 'escape'
                state['phase_start_time'] = current_time
                logging.info(f"🔄 【Short Skate】{agent_id}: Turn Cold→Escape (当前{current_heading:.1f}°)")
        
        # 阶段3: 加速逃离（15秒）
        if state['phase'] == 'escape':
            escape_duration = 15.0
            if phase_time < escape_duration:
                if env.current_step % 30 == 0:
                    logging.info(f"🔄 [Short Skate-Escape] {agent_id} 加速逃离 航向{current_heading:.1f}° 高度{current_alt:.0f}m")
                return 7, 8, 5  # 保持航向+加速
            else:
                # 完成Short Skate，清理状态
                del self.maneuver_states[agent_id]
                logging.info(f"✅ 【Short Skate】{agent_id}完成返航机动 (最终航向{current_heading:.1f}°)")
                return 7, 8, 5  # 保持航向+加速
        
        return 7, 8, 3
    
    def _turn_to_heading(self, env, agent_id: str, target_heading: float, speed_cmd: int = 3) -> tuple:
        """
        转向指定绝对航向
        
        Args:
            target_heading: 目标航向（绝对值，0-360°）
            speed_cmd: 速度指令（默认保持）
        """
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 计算航向差（-180 to +180）
        heading_diff = ((target_heading - current_heading + 180) % 360) - 180
        
        # ✅ 关键修复：渐进式转向，避免90度固定转向导致转圈
        # 每次最多转60度，分多次完成大角度转向
        if abs(heading_diff) < 5.0:
            heading_cmd = 8  # 保持航向
        elif heading_diff > 0:  # 需要右转
            if abs(heading_diff) > 60:
                heading_cmd = 12  # 右转60°（而非90°）
            elif abs(heading_diff) > 40:
                heading_cmd = 11  # 右转45°
            elif abs(heading_diff) > 20:
                heading_cmd = 10  # 右转30°
            else:
                heading_cmd = 9   # 右转15°
        else:  # 需要左转
            if abs(heading_diff) > 60:
                heading_cmd = 4   # 左转60°（而非90°）
            elif abs(heading_diff) > 40:
                heading_cmd = 5   # 左转45°
            elif abs(heading_diff) > 20:
                heading_cmd = 5   # 左侧Crank -45°
            else:
                heading_cmd = 6   # 左转30°
        
        return 7, heading_cmd, speed_cmd
    
    def _get_enemy_bearing(self, env, agent_id: str) -> float:
        """
        计算敌机方位角
        
        Returns:
            敌机方位角（度，0-360°）
        """
        try:
            # 获取我机位置
            my_pos = env.agents[agent_id].get_position()
            
            # 获取敌机位置（使用固定目标分配）
            enemy_id = get_target_with_fallback(agent_id, env)
            if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                enemy_pos = env.agents[enemy_id].get_position()
            else:
                # 没有敌机，返回当前航向
                return env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 计算方位角
            delta_x = enemy_pos[0] - my_pos[0]
            delta_y = enemy_pos[1] - my_pos[1]
            bearing_rad = np.arctan2(delta_y, delta_x)
            bearing_deg = np.rad2deg(bearing_rad)
            
            # 转换为0-360°
            bearing_deg = (90 - bearing_deg) % 360.0  # 从数学坐标系转为航空坐标系
            
            return bearing_deg
        except Exception as e:
            logging.warning(f"计算敌机方位失败: {e}")
            return env.agents[agent_id].get_property_value(c.attitude_psi_deg)
    
    def _calculate_dynamic_crank_angle(self, env, agent_id: str, base_angle: float = 35.0) -> float:
        """
        动态计算Crank角度 - 根据威胁和距离调整
        
        Args:
            base_angle: 基础Crank角度（默认35°）
        
        Returns:
            调整后的Crank角度 (15°-45°)
        """
        try:
            # 1. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            threat_level = threat_info.get('lead_threat', 0.5)  # 默认0.5
            
            # 2. 获取距离
            distance = self._calculate_distance(env)
            
            # 3. 根据威胁调整角度
            if threat_level > 0.7:
                angle_factor = 0.5  # 高威胁：减小角度
            elif threat_level > 0.4:
                angle_factor = 0.75  # 中威胁：适中角度
            else:
                angle_factor = 1.0  # 低威胁：标准角度
            
            # 4. 根据距离调整角度
            if distance < 60000:  # <60km
                distance_factor = 0.6  # 接近发射区：减小角度
            elif distance < 80000:  # <80km
                distance_factor = 0.8  # 中距离：适中角度
            else:
                distance_factor = 1.0  # 远距离：标准角度
            
            # 5. 计算最终角度
            final_angle = base_angle * angle_factor * distance_factor
            
            # 6. 限制角度范围（15°-45°）
            final_angle = max(15.0, min(45.0, final_angle))
            
            return final_angle
        except Exception as e:
            logging.warning(f"动态Crank角度计算失败: {e}，使用默认{base_angle}°")
            return base_angle
    
    def _execute_drag_shoot(self, env, agent_id: str) -> tuple:
        """
        战术1: 拖曳射击 (DRAG_SHOOT) - 严格按照原战术模板
        核心思想: 长机诱敌，僚机射击
        """
        is_lead = agent_id.endswith('100')
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_time = env.current_step * env.time_interval
        
        # 检查Short Skate状态（使用short_skate_states）
        if agent_id in self.short_skate_states:
            return self._execute_short_skate_precise(env, agent_id, current_time)
        
        # 长机动作序列
        if is_lead:
            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    is_lead = agent_id.endswith('100')
                    direction = 'left' if is_lead else 'right'
                    return self._execute_tactical_crank(env, agent_id, direction, climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time)
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time)
        
        # 僚机动作序列（独立阶段判断）- 使用固定目标
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            if wingman_phase == TacticalPhase.NLT_MELD:
                # 初期小角度右转，缩短Crank时间
                return self._maintain_heading_precise(env, agent_id, 35.0)
            elif wingman_phase == TacticalPhase.MELD_MTR:
                # 快速回正到0°
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.MTR_LR:
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 350.0)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time)
                else:
                    return self._maintain_heading_precise(env, agent_id, 350.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time)
    
    def _execute_front_back(self, env, agent_id: str) -> tuple:
        """
        战术4: 前后攻击 (Front-Back Attack) - 严格按照原战术模板
        核心思想: 僚机藏在长机后方3海里，形成一字型纵队，依顺序发射导弹
        
        长机动作序列：
        - NLT-MELD, MELD-MTR, MTR-LR, LR-TR: 保持0°航向平飞
        - TR-DOR: 发射后Short Skate左侧返航
        - DOR-DR及之后: Short Skate返航到180°
        
        僚机动作序列（独立阶段判断）：
        - NLT-MELD, MELD-MTR: 保持0°航向（藏在长机后3海里）
        - MTR-LR, LR-TR: 保持0°航向
        - TR-DOR: 发射后Short Skate右侧返航
        - DOR-DR及之后: Short Skate返航到180°
        
        关键：僚机全程保持0°航向，通过距离滞后自然形成一字型
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        # 前后攻击战术使用专用配置文件，僚机已在正确位置
        
        # 检查Short Skate状态
        if agent_id in self.short_skate_states:
            # 长机左转，僚机右转
            skate_direction = 'left' if is_lead else 'right'
            return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列（前机）
        if is_lead:
            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                # 长机初期加速（缩短时间，减少航向偏离）
                if current_time < 4.0:
                    # 前4秒加速（快速建立速度差）
                    return 7, 8, 4  # 保持高度，保持航向，加速
                else:
                    # 之后恢复正常速度平飞
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 左侧返航
                    return self._execute_short_skate_precise(env, agent_id, current_time, 'left')
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, 'left')
        
        # 僚机动作序列（后机，独立阶段判断）- 使用固定目标
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            # 僚机主动建立和保持后方队形（藏在长机后方3海里）
            if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
                # 建立后方队形：主动调整速度和航向
                return self._establish_rear_formation(env, agent_id, current_time)
            elif wingman_phase == TacticalPhase.MTR_LR:
                # 保持后方队形：跟随长机 - 优先检查是否刚完成Crank
                if hasattr(self, 'wingman_crank_state') and self.wingman_crank_state.get("completed", False):
                    # 刚完成一字型，继续使用establish逻辑保持队形
                    recent_completion = current_time - self.wingman_crank_state.get("completed_time", 0) < 60.0
                    if recent_completion:
                        logging.info(f"🔄 [{agent_id}]刚完成一字型，继续使用establish逻辑保持队形")
                        return self._establish_rear_formation(env, agent_id, current_time)
                # 正常保持后方队形
                return self._maintain_rear_formation(env, agent_id)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或保持队形
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    # 平飞保持后方队形，完成中制导
                    if hasattr(self, 'wingman_crank_state') and self.wingman_crank_state.get("completed", False):
                        recent_completion = current_time - self.wingman_crank_state.get("completed_time", 0) < 60.0
                        if recent_completion:
                            return self._establish_rear_formation(env, agent_id, current_time)
                    return self._maintain_rear_formation(env, agent_id)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 右侧返航
                    return self._execute_short_skate_precise(env, agent_id, current_time, 'right')
                else:
                    # 继续保持后方队形 - 优先使用establish逻辑
                    if hasattr(self, 'wingman_crank_state') and self.wingman_crank_state.get("completed", False):
                        recent_completion = current_time - self.wingman_crank_state.get("completed_time", 0) < 60.0
                        if recent_completion:
                            return self._establish_rear_formation(env, agent_id, current_time)
                    return self._maintain_rear_formation(env, agent_id)
            else:
                # 其他阶段 - 检查是否刚完成一字型，避免立即返航
                if hasattr(self, 'wingman_crank_state') and self.wingman_crank_state.get("completed", False):
                    recent_completion = current_time - self.wingman_crank_state.get("completed_time", 0) < 60.0
                    if recent_completion:
                        logging.info(f"🔄 [{agent_id}]刚完成一字型，继续保持队形而非返航")
                        return self._establish_rear_formation(env, agent_id, current_time)
                return self._execute_short_skate_precise(env, agent_id, current_time, 'right')
    
    def _execute_side_by_side(self, env, agent_id: str) -> tuple:
        """
        战术5: 并排射击 (Side-by-Side) - 严格按照原战术模板
        核心思想: 双机保持队形，同时发射导弹
        Short Skate方向: 长机左转，僚机右转（相反方向）
        """
        is_lead = agent_id.endswith('100')
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        current_time = env.current_step * env.time_interval
        
        # 确定Short Skate方向：长机左转，僚机右转
        skate_direction = 'left' if is_lead else 'right'
        
        # 检查Short Skate状态（使用short_skate_states）
        if agent_id in self.short_skate_states:
            return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列
        if is_lead:
            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段继续保持航向，完成中制导
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：继续制导15秒后再规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                # ✅ 修复：延长制导时间到15秒，确保导弹获得足够制导支持
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    # 继续保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 僚机动作序列（独立阶段判断）- 使用固定目标
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                lead_launch = self.last_missile_launch_time.get('A0100', -999)
                
                # ✅ 僚机发射逻辑：在长机发射后5秒自动发射
                if last_launch < 0:  # 僚机还未发射
                    if lead_launch > 0 and (current_time - lead_launch) >= 5.0:
                        # 长机已发射超过5秒，僚机发射
                        self.missile_launched[agent_id] = True
                        logging.info(f"🚀 [僚机跟随发射] {agent_id} 在长机发射{current_time - lead_launch:.1f}秒后准备发射")
                    elif lead_launch < 0:
                        # 长机还未发射，僚机也标记准备发射（并排射击战术）
                        self.missile_launched[agent_id] = True
                
                # LR阶段继续保持航向，完成中制导
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：继续制导15秒后再规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                # ✅ 修复：延长制导时间到15秒，确保导弹获得足够制导支持
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    # 继续保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
    
    def _execute_high_low_attack(self, env, agent_id: str) -> tuple:
        """
        战术3: 上下夹击 (High-Low Attack) - 严格按照原战术模板
        核心思想: 僚机高空，长机低空，利用高度差创造立体攻击
        
        长机动作序列：
        - NLT-MELD, MELD-MTR, MTR-LR, LR-TR: 保持低空（6000m）0°航向
        - TR-DOR: 发射后Short Skate左侧返航
        - DOR-DR及之后: Short Skate返航到180°
        
        僚机动作序列（独立阶段判断）：
        - NLT-MELD: 保持0°航向
        - MELD-MTR: 爬升到高空（8000m）保持0°航向
        - MTR-LR, LR-TR: 保持高空0°航向
        - TR-DOR: 俯冲发射后Short Skate右侧返航
        - DOR-DR及之后: Short Skate返航到180°
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        current_alt = env.agents[agent_id].get_position()[2]
        
        # 检查Short Skate状态
        if agent_id in self.short_skate_states:
            # 长机左转，僚机右转
            skate_direction = 'left' if is_lead else 'right'
            return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作序列（低空）
        if is_lead:
            # 使用vertical_split设置的目标高度，如果没有则使用默认6000m
            target_altitude = self.vertical_split_targets.get(agent_id, 6096.0)

            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                # 检查是否需要调整高度到目标
                alt_diff = target_altitude - current_alt
                if abs(alt_diff) > 200:
                    # 需要调整高度，计算合理的爬升/下降量
                    if alt_diff > 1000:
                        alt_cmd_value = 1000
                    elif alt_diff > 500:
                        alt_cmd_value = 500
                    elif alt_diff > 200:
                        alt_cmd_value = 300
                    elif alt_diff < -1000:
                        alt_cmd_value = -1000
                    elif alt_diff < -500:
                        alt_cmd_value = -500
                    elif alt_diff < -200:
                        alt_cmd_value = -300
                    else:
                        alt_cmd_value = alt_diff

                    alt_cmd = self._convert_altitude_to_index(alt_cmd_value)
                    return alt_cmd, 8, 3  # 保持航向和速度
                else:
                    # 已到达目标高度，保持0°航向
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time, 'left')
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, 'left')
        
        # 僚机动作序列（高空，独立阶段判断）- 使用固定目标
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase

            # ✅ 修复1.3: 使用vertical_split设置的目标高度，如果没有则使用默认9000m (符合8-10km定义)
            target_altitude = self.vertical_split_targets.get(agent_id, 9144.0)  # 9144m ≈ 9km (修改前: 8000)

            # 【修复9】检查横向分离距离，避免超出CAP边界
            # 计算与长机的横向分离（Y方向）
            lead_pos = env.agents['A0100'].get_position()
            wing_pos = env.agents[agent_id].get_position()
            lateral_separation = abs(wing_pos[1] - lead_pos[1])  # Y方向距离

            # 目标横向分离：5km（根据项目说明.md）
            target_lateral_separation = 5000

            # 检查是否接近CAP边界（Y=±60km）
            wing_y = wing_pos[1]
            approaching_boundary = abs(wing_y) > 50000  # 距离边界10km以内

            # 决定航向偏转角度
            if approaching_boundary:
                # 接近边界，强制返回中心（左偏）
                heading_offset = -10.0
                if env.current_step % 200 == 0:
                    logging.warning(f"⚠️ [{agent_id}] 接近CAP边界(Y={wing_y/1000:.1f}km)，强制左偏返回")
            elif lateral_separation >= target_lateral_separation:
                # 已达到目标分离，保持0°航向（不再右偏）
                heading_offset = 0.0
            else:
                # 未达到目标分离，继续右偏
                heading_offset = 10.0

            if wingman_phase == TacticalPhase.NLT_MELD:
                # 建立横向分离，根据当前分离距离调整航向
                return self._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase == TacticalPhase.MELD_MTR:
                # 爬升到目标高度，同时调整横向分离
                alt_diff = target_altitude - current_alt
                if abs(alt_diff) > 200:
                    # 需要调整高度，计算合理的爬升/下降量
                    # 限制单次调整量，避免过度爬升
                    if alt_diff > 1000:
                        alt_cmd_value = 1000  # 最多爬升1000m
                    elif alt_diff > 500:
                        alt_cmd_value = 500
                    elif alt_diff > 200:
                        alt_cmd_value = 300
                    elif alt_diff < -1000:
                        alt_cmd_value = -1000  # 最多下降1000m
                    elif alt_diff < -500:
                        alt_cmd_value = -500
                    elif alt_diff < -200:
                        alt_cmd_value = -300
                    else:
                        alt_cmd_value = alt_diff

                    alt_cmd = self._convert_altitude_to_index(alt_cmd_value)
                    hdg_cmd = self._convert_heading_to_index(np.deg2rad(heading_offset))
                    return alt_cmd, hdg_cmd, 3
                else:
                    # 已到达高度，根据横向分离调整航向
                    return self._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase == TacticalPhase.MTR_LR:
                # 保持高空，根据横向分离调整航向
                return self._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    # 平飞根据横向分离调整航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, heading_offset)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 俯冲后右侧返航
                    return self._execute_short_skate_precise(env, agent_id, current_time, 'right')
                else:
                    # 俯冲攻击：下降+保持航向+加速
                    return 5, 8, 4  # 下降-500m, 保持航向, 加速+50m/s
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, 'right')
    
    def _execute_pincer_attack(self, env, agent_id: str) -> tuple:
        """
        战术2: 钳形攻势 (Pincer Attack) - 严格按照原战术模板
        核心思想: 双机从两侧包夹敌机，发射后内侧收拢返航
        
        原战术模板动作序列（90-81km展开，81-45km收拢）：
        - NLT-MELD (90-81km): 快速展开（长机315°=-45°，僚机45°）
        - MELD-MTR (81-45km): 收拢到0°并保持（形成钳形夹击姿态）
        - MTR-LR, LR-TR: 保持0°，准备发射
        - TR-DOR: 内侧Short Skate返航
        """
        is_lead = agent_id.endswith('100')
        current_time = env.current_step * env.time_interval
        
        # 确定内侧Short Skate方向：长机右转，僚机左转（与Crank方向相反）
        skate_direction = 'right' if is_lead else 'left'
        
        # 检查Short Skate状态
        if agent_id in self.short_skate_states:
            return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 长机动作（左侧包抄）
        if is_lead:
            # 使用飞机的独立阶段，而不是全局阶段
            lead_phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            
            # 调试：打印长机的阶段和目标航向
            if env.current_step % 60 == 0:
                logging.info(f"  🎯 [钳形攻势-长机] {agent_id} 阶段:{lead_phase.value}")
            
            # 【修复10】检查横向分离距离，避免超出CAP边界
            lead_pos = env.agents[agent_id].get_position()
            wing_pos = env.agents['A0200'].get_position() if env.agents['A0200'].is_alive else lead_pos
            lateral_separation = abs(wing_pos[1] - lead_pos[1])  # Y方向距离
            target_lateral_separation = 10000  # 钳形攻势目标横向分离：10km

            # 检查是否接近CAP边界（Y=±60km）
            lead_y = lead_pos[1]
            approaching_boundary = abs(lead_y) > 50000  # 距离边界10km以内

            if lead_phase == TacticalPhase.NLT_MELD:
                # ✅ 修复1.1: NLT阶段快速展开到左侧-45°（直接计算指令）
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 315.0  # -45°
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
                
                # 如果已达到目标分离或接近边界，停止左偏
                if approaching_boundary or lateral_separation >= target_lateral_separation:
                    if env.current_step % 60 == 0:
                        logging.info(f"     → 已达到分离，保持315°")
                else:
                    if env.current_step % 60 == 0:
                        logging.info(f"     → 目标航向:315° (展开)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3  # 直接返回指令
            elif lead_phase == TacticalPhase.MELD_MTR:
                # ✅ 关键修复：MELD阶段应该保持-15度分离，而非收拢到0度！
                # 原因：钳形攻势MELD阶段是左右夹攻，需要保持横向分离
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 345.0  # 长机-15°（而非0°）
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 60 == 0:
                    logging.info(f"📍 [MELD分离] 长机目标345°(-15°)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3  # 直接返回指令
            elif lead_phase == TacticalPhase.MTR_LR:
                # MTR-LR阶段：保持0°形成钳形夹击
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # ✅ 关键修复：发射后15秒内，禁止任何转向，保持当前姿态
                if last_launch > 0 and (current_time - last_launch) < 15.0:
                    remaining_time = 15.0 - (current_time - last_launch)
                    if env.current_step % 60 == 0:
                        logging.info(f"🎯 [{agent_id}] 导弹制导保护: 剩余{remaining_time:.1f}秒，保持当前姿态")
                    return 7, 8, 3  # 保持高度、航向、中速

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'left', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：继续制导15秒后再规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                # ✅ 修复：延长制导时间到15秒，确保导弹获得足够制导支持
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    # 内侧返航：右转（与左侧Crank相反）
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    # 继续保持航向，完成中制导
                    return 7, 8, 3
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 僚机动作（右侧包抄，独立阶段判断）- 使用固定目标
        else:
            target_id = get_target_with_fallback(agent_id, env)
            target_aircraft = env._jsbsims.get(target_id)
            if target_aircraft and target_aircraft.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], target_aircraft)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase

            # 【修复11】检查横向分离距离，避免超出CAP边界
            lead_pos = env.agents['A0100'].get_position() if env.agents['A0100'].is_alive else env.agents[agent_id].get_position()
            wing_pos = env.agents[agent_id].get_position()
            lateral_separation = abs(wing_pos[1] - lead_pos[1])  # Y方向距离
            target_lateral_separation = 10000  # 钳形攻势目标横向分离：10km

            # 检查是否接近CAP边界（Y=±60km）
            wing_y = wing_pos[1]
            approaching_boundary = abs(wing_y) > 50000  # 距离边界10km以内

            if wingman_phase == TacticalPhase.NLT_MELD:
                # ✅ 修复1.1: NLT阶段快速展开到右侧+45°
                # 如果已达到目标分离或接近边界，停止右偏
                if approaching_boundary or lateral_separation >= target_lateral_separation:
                    target_heading = 45.0  # 保持+45°
                else:
                    target_heading = 45.0  # 展开到+45°
                return self._maintain_heading_precise(env, agent_id, target_heading)
            elif wingman_phase == TacticalPhase.MELD_MTR:
                # ✅ 关键修复：MELD阶段应该保持+15度分离，而非收拢到0度！
                # 原因：钳形攻势MELD阶段是左右夹攻，需要保持横向分离
                current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                target_heading = 15.0  # 僚机+15°（而非0°）
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
                if env.current_step % 60 == 0:
                    logging.info(f"📍 [MELD分离] 僚机目标15°(+15°)，当前{current_heading:.1f}°")
                return 7, hdg_idx, 3
            elif wingman_phase == TacticalPhase.MTR_LR:
                # MTR-LR阶段：保持0°形成钳形夹击
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # ✅ 关键修复：发射后15秒内，禁止任何转向，保持当前姿态
                if last_launch > 0 and (current_time - last_launch) < 15.0:
                    remaining_time = 15.0 - (current_time - last_launch)
                    if env.current_step % 60 == 0:
                        logging.info(f"🎯 [{agent_id}] 导弹制导保护: 剩余{remaining_time:.1f}秒，保持当前姿态")
                    return 7, 8, 3  # 保持高度、航向、中速

                # ✅ 优先级2.2修复：根据LR决策执行Crank或平飞
                lr_maneuver = getattr(self, 'lr_maneuver', {}).get(agent_id, 'straight')
                if lr_maneuver == 'crank':
                    # 执行Crank机动（偏转30°保持规避余度）
                    return self._execute_tactical_crank(env, agent_id, 'right', climb=False)
                else:
                    # 平飞保持航向，完成中制导
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：继续制导15秒后再规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                # ✅ 修复：延长制导时间到15秒，确保导弹获得足够制导支持
                if last_launch > 0 and (current_time - last_launch) > 15.0:
                    # 内侧返航：左转（与右侧Crank相反）
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    # 继续保持航向，完成中制导
                    return 7, 8, 3
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
    
    def _execute_tactical_evasion(self, env, agent_id: str) -> tuple:
        """
        战术6: 战术规避 (Tactical Evasion)
        核心思想: 规避敌方导弹威胁，保存实力
        
        触发条件:
        - RWR告警（敌方雷达锁定）
        - 导弹来袭告警
        - 态势不利（威胁度>0.8）
        
        机动序列:
        1. Notch Back（进入多普勒盲区）
        2. 战术爬升/下降（改变高度层）
        3. Short Skate返航
        """
        is_lead = agent_id.endswith('100')
        agent_phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
        
        # 检查是否正在执行机动
        if agent_id in self.maneuver_states:
            maneuver_type = self.maneuver_states[agent_id].get('type')
            direction = 'left' if is_lead else 'right'
            
            if maneuver_type == 'notch_back':
                return self._execute_notch_back(env, agent_id, direction)
            elif maneuver_type == 'short_skate':
                return self._execute_short_skate(env, agent_id, direction)
        
        # 早期阶段：Notch Back机动
        if agent_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR]:
            direction = 'left' if is_lead else 'right'
            return self._execute_notch_back(env, agent_id, direction)
        
        # 中期阶段：战术爬升/下降 + Beam机动
        elif agent_phase in [TacticalPhase.MTR_LR, TacticalPhase.LR_TR]:
            # 长机战术爬升，僚机战术下降（垂直分散）
            if is_lead:
                return 11, 2, 5  # 战术爬升+500m + 左转90° + 大幅加速
            else:
                return 3, 14, 5  # 战术下降-500m + 右转90° + 大幅加速
        
        # TR-DOR阶段：继续制导后Short Skate返航
        elif agent_phase == TacticalPhase.TR_DOR:
            # ✅ 修复：检查导弹制导保护 - 发射后15秒内继续直飞
            current_time = env.current_step * env.time_interval
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) < 15.0:
                # 继续保持航向，为导弹提供制导支持
                return 7, 8, 3  # 保持高度、航向、中速
            
            direction = 'left' if is_lead else 'right'
            return self._execute_short_skate(env, agent_id, direction)
        
        # DOR-DR阶段：执行规避机动（根据DOR决策）
        elif agent_phase == TacticalPhase.DOR_DR:
            # ✅ 修复：检查导弹制导保护 - 发射后15秒内继续直飞
            current_time = env.current_step * env.time_interval
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) < 15.0:
                # 继续保持航向，为导弹提供制导支持
                return 7, 8, 3  # 保持高度、航向、中速
            
            evasion = getattr(self, 'evasion_maneuver', 'SHORT_SKATE')
            if evasion == 'SHORT_SKATE':
                direction = 'left' if is_lead else 'right'
                return self._execute_short_skate(env, agent_id, direction)
            elif evasion == 'NOTCH_BACK':
                direction = 'left' if is_lead else 'right'
                return self._execute_notch_back(env, agent_id, direction)
            else:  # BEAM
                return self._execute_beam_maneuver(env, agent_id)
        
        # DR-MAR阶段：根据DR决策结果决定是否重新进攻或返航
        elif agent_phase == TacticalPhase.DR_MAR:
            # ✅ 修复：检查导弹制导保护 - 发射后15秒内继续直飞
            current_time = env.current_step * env.time_interval
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if last_launch > 0 and (current_time - last_launch) < 15.0:
                # 继续保持航向，为导弹提供制导支持
                return 7, 8, 3  # 保持高度、航向、中速
            
            # ✅ 关键修复：DR_MAR阶段计算返回基地的航向
            # 原因：hdg=16是"当前+180°"而非"转到180°"，会导致震荡
            # 我方基地：X=-60km(南方)，敌方基地：X=+60km(北方)
            pos = env.agents[agent_id].get_position()
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 根据阵营决定返航方向
            if agent_id.startswith('A'):  # 我方，返南
                target_heading = 180.0
            else:  # 敌方，返北
                target_heading = 0.0
            
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
            
            # 降低返航日志频率：每60步（12秒）输出一次
            if not hasattr(self, '_return_log_counter'):
                self._return_log_counter = {}
            if agent_id not in self._return_log_counter:
                self._return_log_counter[agent_id] = 0
            
            self._return_log_counter[agent_id] += 1
            if self._return_log_counter[agent_id] % 60 == 0:
                logging.info(f"🛡️ [DR_MAR返航] {agent_id} 当前{current_heading:.0f}°→基地{target_heading:.0f}° (X={pos[0]/1000:.1f}km)")
            
            return 7, hdg_idx, 5  # 返航 + 大幅加速
        
        # BEYOND_MAR阶段：强制返航
        elif agent_phase == TacticalPhase.BEYOND_MAR:
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            # 根据阵营决定返航方向
            if agent_id.startswith('A'):  # 我方，返南
                target_heading = 180.0
            else:  # 敌方，返北
                target_heading = 0.0
            heading_diff = self._normalize_angle_diff(target_heading - current_heading)
            hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
            return 7, hdg_idx, 5  # 返航 + 大幅加速
        
        else:
            # 默认：平稳返航
            return 7, 8, 3

    def _execute_beam_maneuver(self, env, agent_id: str) -> tuple:
        """
        执行BEAM机动：相对敌机保持±90°侧对
        返回 (alt_cmd, hdg_cmd, vel_cmd)
        """
        try:
            enemy_bearing = self._get_enemy_bearing(env, agent_id)
            is_lead = agent_id.endswith('100')
            # 长机左侧-90°，僚机右侧+90°
            target_heading = (enemy_bearing - 90.0) % 360.0 if is_lead else (enemy_bearing + 90.0) % 360.0
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            heading_diff = ((target_heading - current_heading + 180) % 360) - 180
            hdg_cmd = self._convert_heading_to_index(np.deg2rad(heading_diff))
            # 保持高度，速度保持（必要时可按距离微调）
            return 7, hdg_cmd, 3
        except Exception:
            return 7, 8, 3

    
    # ==================== 前后攻击辅助函数 ====================
    
    def _establish_rear_formation(self, env, agent_id: str, current_time: float) -> tuple:
        """
        僚机建立后方队形（前后攻击战术专用）- 真正的动态调整
        
        目标：僚机藏在长机后方3海里，形成一字型纵队
        
        
        Args:
            env: 环境对象
            agent_id: 僚机ID
            current_time: 当前时间
            
        Returns:
            tuple: (altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
        """
        # 获取长机和僚机
        leader = env._jsbsims.get("A0100")
        wingman = env._jsbsims.get(agent_id)
        
        # 默认指令
        altitude_cmd_id = 7  # 保持高度
        heading_cmd_id = 8   # 保持航向
        velocity_cmd_id = 3  # 保持速度
        
        if not leader or not wingman:
            return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
        
        # 获取位置和航向
        leader_pos = leader.get_position()
        wingman_pos = wingman.get_position()
        leader_x, leader_y, leader_z = leader_pos[0], leader_pos[1], leader_pos[2]
        wingman_x, wingman_y, wingman_z = wingman_pos[0], wingman_pos[1], wingman_pos[2]
        
        leader_heading = np.rad2deg(leader.get_property_value(c.attitude_psi_rad))
        wingman_heading = np.rad2deg(wingman.get_property_value(c.attitude_psi_rad))
        
        # 高度控制 - 放宽阈值，减少频繁调整
        alt_diff = wingman_z - leader_z
        
        # 初始化状态机（如果不存在）
        if not hasattr(self, 'formation_state'):
            self.formation_state = {}
        if agent_id not in self.formation_state:
            self.formation_state[agent_id] = 'LATERAL_ALIGN'
        
        current_state = self.formation_state.get(agent_id, 'LATERAL_ALIGN')
        
        # 根据状态调整高度控制阈值（根治不稳定问题）
        if current_state in ['LATERAL_ALIGN', 'HEADING_CORRECT']:
            # 建立队形初期：大幅放宽高度控制，让僚机专注于横向/航向对齐
            # 转弯时自然会掉高度，频繁纠正会导致不稳定
            alt_threshold = 300  # ±300m才调整
        elif current_state == 'FORMATION_HOLD':
            # 平飞状态：放宽高度控制
            alt_threshold = 200  # ±200m才调整
        else:
            # 纵向调整阶段：正常控制
            alt_threshold = 150  # ±150m才调整
        
        if alt_diff > alt_threshold:
            altitude_cmd_id = 5  # 下降
            logging.warning(f"⚠️ [{agent_id}]高度过高 {alt_diff:.0f}m (阈值{alt_threshold}m) → 下降")
        elif alt_diff < -alt_threshold:
            altitude_cmd_id = 9  # 上升
            logging.warning(f"⚠️ [{agent_id}]高度过低 {alt_diff:.0f}m (阈值{alt_threshold}m) → 上升")
        
        # 计算相对位置
        dx = wingman_x - leader_x  # 正=前方，负=后方
        dy = wingman_y - leader_y  # 正=右侧，负=左侧
        heading_diff = self._normalize_angle_diff(leader_heading - wingman_heading)
        
        # 精确定义目标参数 - 关键：dy容忍度要小，确保一字型精度
        TARGET_DY_TOLERANCE = 200        # 横向误差容忍±200m（严格，保证一字型精度）
        TARGET_DX_MIN = -12000           # 纵向距离-12km（后方12km）
        TARGET_DX_MAX = -7000            # 纵向距离-7km（后方7km）
        TARGET_HEADING_TOLERANCE = 10.0  # 航向误差容忍±10°（减少微调）
        
        # 稳定区域：进入此区域后完全平飞，不做任何调整（放宽容忍度，防止振荡）
        STABLE_DY_RANGE = 3000           # 横向稳定区±3km（放宽，防止频繁重新对齐）
        STABLE_DX_MIN = -15000           # 纵向稳定区-15km到-5km
        STABLE_DX_MAX = -5000
        STABLE_HEADING_RANGE = 30.0      # 航向稳定区±30°
        
        # current_state已经在高度控制部分初始化了
        
        # 状态转换逻辑
        if current_state == 'LATERAL_ALIGN':
            # 阶段1：横向对齐（左侧Crank快速到长机Y坐标）
            if abs(dy) < TARGET_DY_TOLERANCE:
                self.formation_state[agent_id] = 'HEADING_CORRECT'
                logging.info(f"✅ [{agent_id}]横向对齐完成 dy={dy:.0f}m → 进入航向回正")
            else:
                # 关键策略：dy<1500m且heading>0°时右转，防止过头
                if dy > 0:  # 在右侧，需要接近
                    # 右转条件：dy接近目标且航向差>0°（防止右转过头）
                    if dy < 1500 and heading_diff > 0:
                        # dy越小右转越小，防止过冲
                        if dy < 500:
                            heading_cmd_id = 8  # dy很小时保持航向
                        else:
                            heading_cmd_id = 9  # 微右转
                        velocity_cmd_id = 3
                        logging.info(f"✅ [{agent_id}]阶段1-右转对齐: dy={dy:.0f}m hdg={heading_diff:.1f}°")
                    elif dy > 2000:
                        heading_cmd_id = 6  # 大幅左转30°
                        velocity_cmd_id = 3  # 保持速度
                        logging.info(f"🔄 [{agent_id}]阶段1-大幅左转: dy={dy:.0f}m")
                    else:
                        # dy在1500-2000m之间，中幅左转
                        heading_cmd_id = 7  # 中幅左转15°
                        velocity_cmd_id = 3  # 保持速度
                        logging.info(f"🔄 [{agent_id}]阶段1-中幅左转: dy={dy:.0f}m")
                else:  # 在左侧（已经过头）
                    heading_cmd_id = 9  # 右转回来
                    velocity_cmd_id = 3  # 保持速度
                    logging.warning(f"⚠️ [{agent_id}]阶段1-过头右转: dy={dy:.0f}m")
        
        elif current_state == 'HEADING_CORRECT':
            # 阶段2：航向回正（对齐长机航向，减少微调次数）
            if abs(heading_diff) < TARGET_HEADING_TOLERANCE:
                self.formation_state[agent_id] = 'LONGITUDINAL_ADJUST'
                logging.info(f"✅ [{agent_id}]航向对齐完成 hdg_diff={heading_diff:.1f}° → 进入纵向调整")
            else:
                # 根据航向差决定转向（提高微调阈值，减少调整次数）
                if heading_diff > 30.0:
                    heading_cmd_id = 10  # 大幅右转30°
                    logging.info(f"🔄 [{agent_id}]阶段2-大幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff > TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 9  # 微右转15°
                    logging.info(f"🔄 [{agent_id}]阶段2-微幅右转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -30.0:
                    heading_cmd_id = 6  # 大幅左转30°（不应该发生）
                    logging.warning(f"🔄 [{agent_id}]阶段2-异常左转: hdg_diff={heading_diff:.1f}°")
                elif heading_diff < -TARGET_HEADING_TOLERANCE:
                    heading_cmd_id = 7  # 微左转15°（不应该发生）
                    logging.warning(f"🔄 [{agent_id}]阶段2-异常微左转: hdg_diff={heading_diff:.1f}°")
                else:
                    # 在容忍范围内，保持航向不调整
                    heading_cmd_id = 8
                
                velocity_cmd_id = 3  # 保持速度
        
        elif current_state == 'LONGITUDINAL_ADJUST':
            # 阶段3：纵向距离调整 - 只调速度，不调航向（保持一字型）
            # 使用稳定区域判断，更容易进入平飞状态
            in_stable_zone = (STABLE_DX_MIN <= dx <= STABLE_DX_MAX and 
                            abs(dy) < STABLE_DY_RANGE and 
                            abs(heading_diff) < STABLE_HEADING_RANGE)
            
            if in_stable_zone:
                self.formation_state[agent_id] = 'FORMATION_HOLD'
                logging.info(f"✅ [{agent_id}]进入稳定区域 dx={dx/1000:.2f}km dy={dy/1000:.2f}km → 开始平飞")
            else:
                # 关键：保持航向，只调速度！
                heading_cmd_id = 8  # 保持航向
                velocity_cmd_id = 3  # 默认保持速度
                
                # 只调整纵向距离（速度）
                if dx > 0:  # 在前方，减速
                    velocity_cmd_id = 2
                elif dx > -5000:  # 太靠前（后方<5km），减速
                    velocity_cmd_id = 2
                elif dx < -15000:  # 太靠后（后方>15km），加速
                    velocity_cmd_id = 4
                # 否则保持速度，让它自然收敛
                
                logging.info(f"🔄 [{agent_id}]阶段3-只调速度: dx={dx/1000:.2f}km dy={dy/1000:.2f}km vel={velocity_cmd_id}")
        
        elif current_state == 'FORMATION_HOLD':
            # 阶段4：保持队形 - 动态微调航向和速度，保持一字型
            
            # 检查是否严重脱离队形（使用稳定区域判断）
            seriously_off = (abs(dy) > STABLE_DY_RANGE or 
                           dx > STABLE_DX_MAX or 
                           dx < STABLE_DX_MIN or
                           abs(heading_diff) > STABLE_HEADING_RANGE)
            
            if seriously_off:
                self.formation_state[agent_id] = 'LATERAL_ALIGN'
                logging.warning(f"⚠️ [{agent_id}]严重脱离队形 dx={dx/1000:.2f}km dy={dy:.0f}m hdg={heading_diff:.1f}° → 重新对齐")
            else:
                # 在稳定区域内：动态微调保持一字型
                heading_cmd_id = 8  # 默认保持航向
                velocity_cmd_id = 3  # 默认保持速度
                
                # 横向微调：dy偏差>500m时调整航向
                if dy < -500:  # 在左侧，微右转
                    heading_cmd_id = 9
                elif dy > 500:  # 在右侧，微左转
                    heading_cmd_id = 7
                
                # 速度微调：根据长机速度动态调整
                leader_velocity = leader.get_property_value(c.velocities_u_fps)
                wingman_velocity = wingman.get_property_value(c.velocities_u_fps)
                speed_diff = leader_velocity - wingman_velocity
                
                # 纵向距离太近(<6km)且速度慢于长机，加速
                if dx > -6000 and speed_diff > 10:
                    velocity_cmd_id = 4  # 加速
                # 纵向距离太远(>12km)且速度快于长机，减速
                elif dx < -12000 and speed_diff < -10:
                    velocity_cmd_id = 2  # 减速
                # 否则根据速度差微调
                elif speed_diff > 30:  # 慢太多，加速
                    velocity_cmd_id = 4
                elif speed_diff < -30:  # 快太多，减速
                    velocity_cmd_id = 2
                
                if env.current_step % 60 == 0:
                    logging.info(f"✅ [{agent_id}]队形微调: dx={dx/1000:.2f}km dy={dy/1000:.2f}km vel_diff={speed_diff:.0f}fps hdg={heading_cmd_id} vel={velocity_cmd_id}")
        
        return altitude_cmd_id, heading_cmd_id, velocity_cmd_id
    
    def _maintain_rear_formation(self, env, agent_id: str):
        """僚机保持后方队形 - 前后攻击专用"""
        # 简化版本，直接调用_establish_rear_formation
        current_time = env.current_step * env.time_interval
        return self._establish_rear_formation(env, agent_id, current_time)
    
    # ==================== 旧代码已删除 ====================
    
    # ==================== 前后攻击辅助函数结束 ====================
    
    def _execute_tactical_turn(self, env, agent_id: str) -> tuple:
        """
        战朧7: 战术转弯 (Tactical Turn)
        核心思想: 大角度转向改变战术态势
        
        应用场景:
        - 态势不利时脱离
        - 重新占位
        - 规避威胁区域
        - 调整攻击角度
        
        机动序列:
        1. 大角度转弯（90°-180°）
        2. 加速脱离
        3. 重新评估态势
        """
        is_lead = agent_id.endswith('100')
        agent_phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
        
        # 检查是否正在执行Short Skate（只检查type='short_skate'的状态）
        if agent_id in self.maneuver_states and self.maneuver_states[agent_id].get('type') == 'short_skate':
            direction = 'left' if is_lead else 'right'
            return self._execute_short_skate(env, agent_id, direction)
        
        # 战术回转：返回基地
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        # 根据阵营决定返航方向
        if agent_id.startswith('A'):  # 我方，返南
            target_heading = 180.0
        else:  # 敌方，返北
            target_heading = 0.0
        heading_diff = self._normalize_angle_diff(target_heading - current_heading)
        hdg_idx = self._convert_heading_to_index(np.deg2rad(heading_diff))
        return 7, hdg_idx, 5  # 高度保持 + 返基地 + 大幅加速
    
    def _update_integrated_tactical_system(self, env, agent_id: str):
        """
        更新集成战术系统
        进行态势评估、意图识别和威胁决策
        """
        try:
            # 获取当前阶段
            phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            
            # 找到我方所有存活飞机
            my_aircraft_list = [env.agents[aid] for aid in env.agents if aid.startswith('A') and env.agents[aid].is_alive]
            
            # 对每个敌方飞机进行评估
            for enemy_id in env.agents:
                if not enemy_id.startswith('B'):
                    continue
                
                if not env.agents[enemy_id].is_alive:
                    continue
                
                # 1. 更新态势评估
                self.integrated_system.update_situation_assessment(
                    env,
                    agent_id,
                    enemy_id,
                    phase
                )
                
                # 2. 更新意图识别（每个敌机只需识别一次）
                if agent_id == "A0100":  # 由长机统一识别
                    self.integrated_system.update_intent_recognition(
                        env,
                        enemy_id,
                        my_aircraft_list
                    )
                
                # 3. 制定威胁决策
                decision = self.integrated_system.make_tactical_decision(
                    env,
                    agent_id,
                    enemy_id,
                    phase
                )
                
                # 4. 根据决策调整行为（可选：在这里覆盖战术选择）
                # 目前只是评估和记录，实际执行仍由原有战术系统处理
                
        except Exception as e:
            logging.error(f"集成战术系统更新失败 {agent_id}: {e}")
    
    def _apply_speed_safety_check(self, env, agent_id, alt_cmd, hdg_cmd, vel_cmd):
        """
        【P0-2修复】速度安全检查 - 防止失速坠毁
        
        Args:
            env: 环境
            agent_id: 飞机ID
            alt_cmd, hdg_cmd, vel_cmd: 原始指令索引
            
        Returns:
            修正后的指令索引 (alt_cmd, hdg_cmd, vel_cmd)
        """
        try:
            aircraft = env.agents[agent_id]
            if not aircraft.is_alive:
                return alt_cmd, hdg_cmd, vel_cmd
            
            # 获取当前状态
            current_speed = np.linalg.norm(aircraft.get_velocity())  # m/s
            current_alt = aircraft.get_position()[2]  # m
            
            # 失速保护阈值（降低以允许战术机动）
            STALL_SPEED_LOW = 120  # 432 km/h - 只在真正危险时介入
            STALL_SPEED_MED = 140  # 504 km/h - 允许高空减速转弯
            HIGH_ALT_THRESHOLD = 12000  # 12km - 提高高空判断阈值
            CRITICAL_ALT_THRESHOLD = 10000  # 10km
            
            original_cmd = (alt_cmd, hdg_cmd, vel_cmd)
            safety_triggered = False
            
            # 情况1: 低速保护（速度 < 150 m/s）
            if current_speed < STALL_SPEED_LOW:
                vel_cmd = max(vel_cmd, 4)  # 强制加速（最少vel_cmd=4）
                safety_triggered = True
                if current_alt > CRITICAL_ALT_THRESHOLD:
                    alt_cmd = min(alt_cmd, 7)  # 禁止继续爬升
                logging.warning(f"⚠️ [{agent_id}] 低速警告({current_speed:.1f}m/s)! 强制加速")
            
            # 情况2: 高空低速保护（高度 > 10km 且速度 < 170 m/s）
            elif current_alt > HIGH_ALT_THRESHOLD and current_speed < STALL_SPEED_MED:
                alt_cmd = min(alt_cmd, 5)  # 限制爬升或强制下降
                vel_cmd = max(vel_cmd, 4)  # 强制加速
                safety_triggered = True
                logging.warning(f"⚠️ [{agent_id}] 高空低速({current_alt/1000:.1f}km, {current_speed:.1f}m/s)! "
                              f"限制高度变化并加速")
            
            # 情况3: 极高空保护（高度 > 15km）
            elif current_alt > 15000:
                if current_speed < 180:
                    # 高度过高且速度不足，强制下降换取速度
                    alt_cmd = 3  # 强制下降-500m
                    vel_cmd = 5  # 全速加速
                    safety_triggered = True
                    logging.warning(f"⚠️ [{agent_id}] 极高空低速! 强制下降换速度")
                else:
                    # 高度过高但速度可以，逐步下降
                    alt_cmd = min(alt_cmd, 6)  # 限制爬升
            
            # 情况4: 低高度保护（降低到2.5km，允许战术性低空飞行）
            elif current_alt < 2500:
                alt_cmd = max(alt_cmd, 7)  # 禁止下降
                if current_alt < 2000:
                    alt_cmd = 11  # 强制爬升+500m
                    safety_triggered = True
                    logging.warning(f"⚠️ [{agent_id}] 低高度警告({current_alt/1000:.1f}km)! 强制爬升")
            
            # 只在触发保护时输出详细日志（降低频率）
            if safety_triggered and env.current_step % 120 == 0:  # 减少日志频率
                logging.info(f"🛡️ [{agent_id}] 速度保护: {original_cmd} → {(alt_cmd, hdg_cmd, vel_cmd)} "
                           f"(速度{current_speed:.1f}m/s, 高度{current_alt/1000:.1f}km)")
            
            return alt_cmd, hdg_cmd, vel_cmd
            
        except Exception as e:
            logging.error(f"速度安全检查失败 {agent_id}: {e}")
            return alt_cmd, hdg_cmd, vel_cmd
    
    def normalize_action(self, env, agent_id, action):
        """
        战术模板核心方法 - 动作归一化
        完全按照drag_shoot_tactical_task.py的架构实现
        注意：action参数未使用，因为使用内部战术逻辑生成动作
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
        
        # 更新战术阶段
        self._update_tactical_phase(env, agent_id)
        
        # === 新增：集成战术系统更新 ===
        if agent_id.startswith('A'):  # 只对我方飞机进行完整决策
            self._update_integrated_tactical_system(env, agent_id)
        
        # 获取战术指令索引（已包含全局安全检查）
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)
        
        # 关键调试日志：每120步（24秒）打印一次指令
        if env.current_step % 120 == 0 and agent_id.startswith('A'):
            alt_change = self.norm_delta_altitude[altitude_cmd_id] * 1000
            hdg_change = np.rad2deg(self.norm_delta_heading[heading_cmd_id])
            vel_change = self.norm_delta_velocity[velocity_cmd_id] * 100
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            logging.info(f"🔧 [{agent_id}] 阶段:{phase.value} 当前航向:{current_heading:.1f}° 指令:alt={altitude_cmd_id},hdg={heading_cmd_id}({hdg_change:+.1f}°),vel={velocity_cmd_id}")
        
        # ✅ 敌方导弹发射检查（由enemy_ai内部自动处理，无需外部调用）
        # 敌方AI系统会在get_enemy_command_indices中自动处理导弹发射
        
        # 使用baseline模型生成底层控制
        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
    
    def _execute_beam_maneuver(self, env, agent_id: str) -> tuple:
        """
        执行Beam机动 - 三九机动，将敌机置于自身3/9位置
        目标：通过转向使敌机位于相对方位角的90°位置（左或右侧）
        """
        # ✅ 关键修复：导弹制导保护 - 发射后15秒内禁止BEAM机动
        current_time = env.current_step * env.time_interval
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if last_launch > 0 and (current_time - last_launch) < 15.0:
            # 继续保持航向，为导弹提供制导支持
            remaining_time = 15.0 - (current_time - last_launch)
            if env.current_step % 60 == 0:  # 每12秒输出一次
                logging.info(f"🎯 [{agent_id}] 导弹制导保护: 剩余{remaining_time:.1f}秒，保持直飞")
            return 7, 8, 3  # 保持高度、航向、中速
        
        # 获取最近敌机的方位角
        enemy_bearing = self._get_enemy_bearing(env, agent_id)
        if enemy_bearing is None:
            return 7, 8, 3  # 无敌机，保持当前姿态
        
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        
        # 计算目标航向：使敌机位于3点或9点方向（左右侧90°）
        # 选择较近的方向（左转或右转）
        beam_left = (enemy_bearing - 90) % 360   # 3点方向（左侧90°）
        beam_right = (enemy_bearing + 90) % 360  # 9点方向（右侧90°）
        
        # 选择较近的目标航向
        diff_left = abs(self._normalize_angle_diff(beam_left - current_heading))
        diff_right = abs(self._normalize_angle_diff(beam_right - current_heading))
        
        if diff_left < diff_right:
            target_heading = beam_left
            beam_direction = "3点方向"
        else:
            target_heading = beam_right
            beam_direction = "9点方向"
        
        # ✅ 关键修复：限制BEAM转向角度，防止东西方向偏离
        heading_diff = self._normalize_angle_diff(target_heading - current_heading)
        if abs(heading_diff) > 30:  # 超过30度，限制到±30度
            if heading_diff > 0:
                target_heading = (current_heading + 30) % 360
            else:
                target_heading = (current_heading - 30) % 360
            if env.current_step % 60 == 0:
                logging.warning(f"⚠️ [{agent_id}] BEAM转向限制: 原目标{heading_diff:.1f}°→限制±30°")
        
        # ✅ 关键修复1：BEAM机动持续时间控制（5秒执行+10秒冷却）
        if not hasattr(self, 'beam_maneuver_state'):
            self.beam_maneuver_state = {}
        
        current_time = env.current_step * env.time_interval
        
        # 检查是否在冷却期
        if agent_id in self.beam_maneuver_state:
            state = self.beam_maneuver_state[agent_id]
            if 'start_time' in state:
                elapsed = current_time - state['start_time']
                
                # 阶段1：执行BEAM（0-5秒）
                if elapsed < 5.0:
                    # 继续执行BEAM转向
                    if env.current_step % 60 == 0:
                        logging.info(f"📍 [BEAM执行] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}° (剩{5.0-elapsed:.1f}s)")
                    return self._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
                
                # 阶段2：冷却期（5-15秒）- 保持航向不动
                elif elapsed < 15.0:
                    if env.current_step % 60 == 0:
                        logging.info(f"⏸️ [BEAM冷却] {agent_id} 保持航向 (剩{15.0-elapsed:.1f}s)")
                    return 7, 8, 3  # 保持当前姿态
                
                # 阶段3：冷却完成，开始新一轮BEAM
                else:
                    state['start_time'] = current_time
                    logging.info(f"📍 [BEAM启动] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}°")
                    return self._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
        else:
            # 首次启动BEAM
            self.beam_maneuver_state[agent_id] = {'start_time': current_time}
            logging.info(f"📍 [BEAM启动] {agent_id} 敌机{enemy_bearing:.1f}° → {beam_direction} {target_heading:.1f}°")
            return self._turn_to_heading(env, agent_id, target_heading, speed_cmd=3)
    
    def _execute_tactical_climb(self, env, agent_id: str, direction='left') -> tuple:
        """
        执行战术爬升 - crank+爬升+反向crank+平飞，机动结束时方位角不变
        四阶段：初始Crank → 爬升+反向Crank → 平飞 → 完成
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_alt = env.agents[agent_id].get_position()[2]
        
        # 初始化战术爬升状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'tactical_climb':
            crank_angle = -30.0 if direction == 'left' else 30.0
            target_alt = current_alt + 1000  # 爬升1000m
            
            self.maneuver_states[agent_id] = {
                'type': 'tactical_climb',
                'phase': 'initial_crank',
                'start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading,
                'initial_alt': current_alt,
                'target_alt': target_alt,
                'crank_angle': crank_angle
            }
            logging.info(f"📍 [战术爬升] {agent_id}开始战术爬升: {direction}侧crank+爬升{target_alt-current_alt:.0f}m")
        
        state = self.maneuver_states[agent_id]
        elapsed_time = current_time - state['start_time']
        
        # 阶段1: 初始Crank（5秒）
        if state['phase'] == 'initial_crank':
            if elapsed_time < 5.0:
                target_heading = (state['initial_heading'] + state['crank_angle']) % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3  # 保持高度+crank+保持速度
            else:
                state['phase'] = 'climb_reverse_crank'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术爬升] {agent_id}: 初始Crank→爬升+反向Crank")
        
        # 阶段2: 爬升+反向Crank（10秒）
        elif state['phase'] == 'climb_reverse_crank':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 10.0:
                # 爬升控制
                alt_diff = state['target_alt'] - current_alt
                if alt_diff > 200:
                    alt_cmd = 11  # +300m中等爬升
                elif alt_diff > 50:
                    alt_cmd = 9   # +50m微调爬升
                else:
                    alt_cmd = 7   # 保持高度
                
                # 反向Crank控制
                reverse_target = (state['initial_heading'] - state['crank_angle']) % 360
                heading_diff = self._normalize_angle_diff(reverse_target - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                
                return alt_cmd, hdg_cmd, 3
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术爬升] {agent_id}: 爬升+反向Crank→平飞")
        
        # 阶段3: 平飞（5秒）
        elif state['phase'] == 'level_off':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 5.0:
                # 返回初始航向
                heading_diff = self._normalize_angle_diff(state['initial_heading'] - current_heading)
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3  # 保持高度+返回航向+保持速度
            else:
                # 完成战术爬升
                del self.maneuver_states[agent_id]
                logging.info(f"✅ [战术爬升] {agent_id}完成战术爬升机动 (最终航向{current_heading:.1f}°)")
                return 7, 8, 3
        
        return 7, 8, 3
    
    def _execute_tactical_descent(self, env, agent_id: str, direction='left') -> tuple:
        """
        执行战术下降 - crank+下降+反向crank+平飞，机动结束时方位角不变
        四阶段：初始Crank → 下降+反向Crank → 平飞 → 完成
        """
        current_time = env.current_step * env.time_interval
        current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
        current_alt = env.agents[agent_id].get_position()[2]
        
        # 初始化战术下降状态
        if agent_id not in self.maneuver_states or self.maneuver_states[agent_id].get('type') != 'tactical_descent':
            crank_angle = -30.0 if direction == 'left' else 30.0
            target_alt = max(current_alt - 500, 3000)  # 下降500m，但不低于3000m
            
            self.maneuver_states[agent_id] = {
                'type': 'tactical_descent',
                'phase': 'initial_crank',
                'start_time': current_time,
                'direction': direction,
                'initial_heading': current_heading,
                'initial_alt': current_alt,
                'target_alt': target_alt,
                'crank_angle': crank_angle
            }
            logging.info(f"📍 [战术下降] {agent_id}开始战术下降: {direction}侧crank+下降{current_alt-target_alt:.0f}m")
        
        state = self.maneuver_states[agent_id]
        elapsed_time = current_time - state['start_time']
        
        # 阶段1: 初始Crank（5秒）
        if state['phase'] == 'initial_crank':
            if elapsed_time < 5.0:
                target_heading = (state['initial_heading'] + state['crank_angle']) % 360
                heading_diff = self._normalize_angle_diff(target_heading - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3  # 保持高度+crank+保持速度
            else:
                state['phase'] = 'descent_reverse_crank'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术下降] {agent_id}: 初始Crank→下降+反向Crank")
        
        # 阶段2: 下降+反向Crank（10秒）
        elif state['phase'] == 'descent_reverse_crank':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 10.0:
                # 下降控制
                alt_diff = current_alt - state['target_alt']
                if alt_diff > 200:
                    alt_cmd = 3   # -300m中等下降
                elif alt_diff > 50:
                    alt_cmd = 5   # -50m微调下降
                else:
                    alt_cmd = 7   # 保持高度
                
                # 反向Crank控制
                reverse_target = (state['initial_heading'] - state['crank_angle']) % 360
                heading_diff = self._normalize_angle_diff(reverse_target - current_heading)
                
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                
                return alt_cmd, hdg_cmd, 3
            else:
                state['phase'] = 'level_off'
                state['phase_start_time'] = current_time
                logging.info(f"📍 [战术下降] {agent_id}: 下降+反向Crank→平飞")
        
        # 阶段3: 平飞（5秒）
        elif state['phase'] == 'level_off':
            phase_time = current_time - state['phase_start_time']
            if phase_time < 5.0:
                # 返回初始航向
                heading_diff = self._normalize_angle_diff(state['initial_heading'] - current_heading)
                if abs(heading_diff) > 5.0:
                    hdg_cmd = 6 if heading_diff < 0 else 10
                else:
                    hdg_cmd = 8
                return 7, hdg_cmd, 3  # 保持高度+返回航向+保持速度
            else:
                # 完成战术下降
                del self.maneuver_states[agent_id]
                logging.info(f"✅ [战术下降] {agent_id}完成战术下降机动 (最终航向{current_heading:.1f}°)")
                return 7, 8, 3
        
        return 7, 8, 3
