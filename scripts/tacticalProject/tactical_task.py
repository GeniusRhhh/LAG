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

from core import TacticalDecisionManager
from core.threat_evaluator import ThreatEvaluator
from core.intent_predictor import IntentPredictor
from core.decision_table import DecisionTable
from core.tactic_selector_v2 import TacticSelectorV2
from enemy_ai_adapter import EnemyAIAdapter
from utils.data_logger import DataLogger

# 导入雷达系统
import sys
import os
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
        
        # 战术决策管理器
        if decision_manager is None:
            self.decision_manager = TacticalDecisionManager()
        else:
            self.decision_manager = decision_manager
        
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
        # 配置：长轴140km（南北/X方向），短轴70km（东西/Y方向）
        # 红方初始位置：X=-60km（南端），蓝方：X=+60km（北端）
        # 两端半圆在南北两端，直道沿南北分布
        self.cap_boundary = {
            'type': 'racetrack',  # 跑马道类型
            'center_x': 0,        # 中心X坐标
            'center_y': 0,        # 中心Y坐标
            'length': 140000,     # 长轴140km（南北方向）
            'width': 70000,       # 短轴70km（东西方向）
            'z_min': 5000,        # 5km最低高度
            'z_max': 15000,       # 15km最高高度
            # 计算矩形部分和半圆部分（南北向）
            'rect_x_min': -35000,  # 矩形南边界 -35km
            'rect_x_max': 35000,   # 矩形北边界 +35km
            'rect_y_min': -35000,  # 矩形西边界 -35km
            'rect_y_max': 35000,   # 矩形东边界 +35km
            'circle_radius': 35000, # 半圆半径 35km（直径70km）
            'south_circle_center': (-35000, 0),   # 南端半圆圆心（X=-35km）
            'north_circle_center': (35000, 0),    # 北端半圆圆心（X=+35km）
        }
        logging.info(f"✅ CAP区域设置: 南北向跑马道")
        logging.info(f"   长轴: 140km (南北/X方向)")
        logging.info(f"   短轴: 70km (东西/Y方向)")
        logging.info(f"   矩形部分: X=[-35,+35]km Y=[-35,+35]km")
        logging.info(f"   南端半圆: 圆心X=-35km, 半径35km")
        logging.info(f"   北端半圆: 圆心X=+35km, 半径35km")
        logging.info(f"   高度: Z=[5,15]km")
        
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
        self.tactical_distances = {
            'MELD': 100000,  # 100km: 雷达融合&调整编队
            'MTR': 80000,    # 80km: MTR1 跟踪目标&进入作战姿态
            'LR': 78000,     # 78km: LR1 导弹最大发射距离
            'TR': 75000,     # 75km: TR1 中制导结束
            'DOR': 70000,    # 70km: Beam/short skate/Notch back
            'DR': 65000,     # 65km: 重新转入进攻/脱离
            'MTR2': 55000,   # 55km: MTR2 第二次进攻-跟踪目标
            'LR2': 53000,    # 53km: LR2 第二次进攻-发射
            'TR2': 50000,    # 50km: TR2 第二次进攻-规避
            'MAR': 40000,    # 40km: 最小规避距离，强制脱离
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
        
        # ===== 决策系统 =====
        self.threat_evaluator = ThreatEvaluator()
        self.intent_predictor = IntentPredictor()
        self.decision_table = DecisionTable()
        self.tactic_selector = TacticSelectorV2(self.threat_evaluator)
        
        # 我方意图（可配置）
        self.my_intent = 'CONSERVATIVE_CLEAR'  # AGGRESSIVE_CLEAR, CONSERVATIVE_CLEAR, DEFENSIVE
        
        # 当前选定的战术（由决策系统决定）
        self.selected_tactic = None
        
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
            logging.info(f"[调试] 步数: {env.current_step}, 时间: {current_time:.1f}s")
            for agent_id in env.agents.keys():
                if env.agents[agent_id].is_alive:
                    pos = env.agents[agent_id].get_position()
                    vel = env.agents[agent_id].get_velocity()
                    speed = np.linalg.norm(vel)
                    alt = pos[2]
                    heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
                    pitch = env.agents[agent_id].get_property_value(c.attitude_theta_deg)
                    roll = env.agents[agent_id].get_property_value(c.attitude_phi_deg)
                    logging.info(f"[{agent_id}] 位置:({pos[0]:.0f},{pos[1]:.0f},{alt:.0f}m) "
                               f"速度:{speed:.1f}m/s 航向:{heading:.1f}° 俯仰:{pitch:.1f}° 滚转:{roll:.1f}°")
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
            
            # 调试：打印输入给baseline的指令（每60步）
            if env.current_step % 60 == 0 and agent_id.startswith('A'):
                logging.info(f"   → baseline输入: 归一化值 alt={input_obs[0]:.3f} hdg={input_obs[1]:.3f} vel={input_obs[2]:.3f}")
            
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
            
            # 6. 安全检查：低高度时避免下降
            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)  # 不允许下降
                norm_act[3] = max(norm_act[3], 0.8)  # 增加油门
            
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
                continue
            
            # 检查冷却时间
            last_launch = self.last_missile_launch_time.get(agent_id, -999)
            if current_time - last_launch < self.missile_cooldown:
                continue
            
            # 检查导弹数量
            if env.agents[agent_id].num_missiles <= 0:
                logging.warning(f"⚠️ {agent_id} 导弹已用尽")
                continue
            
            # 寻找目标
            target = self._find_best_target(env, agent_id)
            if target is None:
                logging.warning(f"⚠️ {agent_id} 未找到有效目标")
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
            
            # 每60步打印一次雷达状态
            if env.current_step % 60 == 0:
                for agent_id in ["A0100", "A0200"]:
                    if agent_id in env.agents and env.agents[agent_id].is_alive:
                        status = self.radar_manager.friendly_radar_states.get(agent_id, "N/A")
                        targets = self.radar_manager.friendly_radar_targets.get(agent_id, {})
                        logging.info(f"[雷达] {agent_id} 状态:{status} 跟踪目标数:{len(targets)}")
        except Exception as e:
            logging.warning(f"雷达系统更新异常: {e}")
    
    def _calculate_distance(self, env) -> float:
        """计算敌我距离（长机到敌方长机）"""
        try:
            leader_red = env._jsbsims.get("A0100")
            leader_blue = env._jsbsims.get("B0100")
            
            if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
                return float('inf')
            
            pos_red = np.array(leader_red.get_position())
            pos_blue = np.array(leader_blue.get_position())
            distance = np.linalg.norm(pos_red - pos_blue)
            
            return distance
        except Exception as e:
            logging.warning(f"计算距离失败: {e}")
            return float('inf')
    
    def _calculate_distance_between(self, aircraft1, aircraft2) -> float:
        """计算两架飞机之间的距离"""
        try:
            pos1 = np.array(aircraft1.get_position())
            pos2 = np.array(aircraft2.get_position())
            return np.linalg.norm(pos1 - pos2)
        except Exception as e:
            logging.error(f"计算距离错误: {e}")
            return float('inf')
    
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
            self.second_attack = False
        
        # 更新最小距离
        if distance < self.min_distance_reached:
            self.min_distance_reached = distance
        
        # 如果距离曾经到达过DR（<20km），现在又拉大到MTR以上（>80km），说明进入第二次进攻
        if self.min_distance_reached < self.tactical_distances['DR'] and distance > self.tactical_distances['MTR']:
            self.second_attack = True
            logging.info(f"🔄 [第二次进攻] 距离从{self.min_distance_reached/1000:.1f}km拉大到{distance/1000:.1f}km，切换到MTR2阶段")
        
        # 阶段判定（添加调试日志）
        if distance > self.tactical_distances['MELD']:  # > 100km
            phase = TacticalPhase.NLT_MELD
            # if env.current_step % 30 == 0:
            #     logging.warning(f"  → NLT_MELD 距离={distance/1000:.1f}km > {self.tactical_distances['MELD']/1000:.1f}km")
        elif distance > self.tactical_distances['MTR']:  # > 80km and <= 100km
            phase = TacticalPhase.MELD_MTR
            # if env.current_step % 30 == 0:
            #     logging.warning(f"  → MELD_MTR 距离={distance/1000:.1f}km > {self.tactical_distances['MTR']/1000:.1f}km")
        elif distance > self.tactical_distances['LR']:  # > 78km and <= 80km
            # 第二次进攻标记
            if self.second_attack:
                logging.debug(f"   [MTR2阶段] 距离{distance/1000:.1f}km")
            phase = TacticalPhase.MTR_LR
        elif distance > self.tactical_distances['TR']:  # > 75km and <= 78km
            if self.second_attack:
                logging.debug(f"   [LR2阶段] 距离{distance/1000:.1f}km")
            phase = TacticalPhase.LR_TR
        elif distance > self.tactical_distances['DOR']:  # > 70km and <= 75km
            if self.second_attack:
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
            # MTR节点：机动决策（是否撤退）
            self._decide_at_mtr(env, agent_id)
        
        elif phase == TacticalPhase.LR_TR:
            # LR节点：参数决策（Crank或平飞）
            self._decide_at_lr(env, agent_id)
        
        elif phase == TacticalPhase.TR_DOR:
            # TR节点：机动决策（继续进攻或脱离）
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
    
    def _decide_at_nlt(self, env):
        """NLT节点决策：策略决策 + 战术决策"""
        try:
            logging.info("=" * 80)
            logging.info("🎯 [NLT节点] 开始战术决策")
            
            # 1. 态势评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            threat_level = threat_info.get('situation', 'ADVANTAGE')
            logging.info(f"   态势评估: {threat_level} (长机威胁:{threat_info.get('lead_threat', 0.5):.2f}, 僚机威胁:{threat_info.get('wingman_threat', 0.5):.2f})")
            
            # 2. 意图预测
            my_lead = env._jsbsims.get("A0100")
            enemy_lead = env._jsbsims.get("B0100")
            
            if my_lead and enemy_lead:
                enemy_intent = self.intent_predictor.predict_enemy_intent(enemy_lead, my_lead, env)
                enemy_intent_type = self.intent_predictor.classify_intent_type(enemy_intent)
                logging.info(f"   敌机意图: {enemy_intent} -> {enemy_intent_type}")
            else:
                enemy_intent_type = 'ATTACK_TYPE'
                logging.warning("   无法获取敌机信息，默认为攻击类型")
            
            # 3. 查询决策表
            tactics, maneuvers = self.decision_table.query_nlt(
                self.my_intent, threat_level, enemy_intent_type
            )
            logging.info(f"   决策表查询: 我方意图={self.my_intent}")
            logging.info(f"   可用战术: {tactics}")
            
            # 4. 选择战术
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]
            
            self.selected_tactic = self.tactic_selector.select_tactic_from_candidates(
                tactics, my_aircraft, enemy_aircraft, env
            )
            
            logging.info(f"   ✅ 选定战术: {self.selected_tactic}")
            logging.info("=" * 80)
            
            # 记录决策日志
            self.data_logger.log_decision(
                env.current_step * env.time_interval, 'A0100', 'NLT_MELD', 'TACTIC',
                self.selected_tactic, None, f"Threat:{threat_level}, Intent:{enemy_intent_type}")
            
        except Exception as e:
            logging.error(f"❌ NLT决策错误: {e}")
            import traceback
            logging.error(traceback.format_exc())
            self.selected_tactic = 'SIDE_BY_SIDE'
    
    def _decide_at_meld(self, env):
        """MELD节点决策：战术调整"""
        try:
            # 1. 态势评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            threat_level = threat_info.get('situation', 'ADVANTAGE')
            
            # 2. 意图预测
            my_lead = env._jsbsims.get("A0100")
            enemy_lead = env._jsbsims.get("B0100")
            
            if my_lead and enemy_lead:
                enemy_intent = self.intent_predictor.predict_enemy_intent(enemy_lead, my_lead, env)
                enemy_intent_type = self.intent_predictor.classify_intent_type(enemy_intent)
            else:
                enemy_intent_type = 'ATTACK_TYPE'
            
            # 3. 查询决策表
            tactics, maneuvers = self.decision_table.query_meld(
                self.my_intent, threat_level, enemy_intent_type
            )
            
            # 4. 可能调整战术
            my_aircraft = [env._jsbsims.get("A0100"), env._jsbsims.get("A0200")]
            enemy_aircraft = [env._jsbsims.get("B0100"), env._jsbsims.get("B0200")]
            
            new_tactic = self.tactic_selector.select_tactic_from_candidates(
                tactics, my_aircraft, enemy_aircraft, env
            )
            
            if new_tactic != self.selected_tactic:
                logging.info(f"🔄 MELD调整战术: {self.selected_tactic} → {new_tactic}")
                
                # 记录决策日志
                self.data_logger.log_decision(
                    env.current_step * env.time_interval, 'A0100', 'MELD_MTR', 'TACTIC',
                    new_tactic, None, f"Adjusted from {self.selected_tactic}")
                
                self.selected_tactic = new_tactic
            
        except Exception as e:
            logging.error(f"MELD决策错误: {e}")
    
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
            
            # 2. 获取敌机意图
            my_aircraft = env.agents.get(agent_id)
            enemy_id = "B0100" if agent_id.startswith('A') else "A0100"
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
            
        except Exception as e:
            logging.error(f"MTR决策错误: {e}")
    
    def _decide_at_lr(self, env, agent_id: str):
        """
        LR节点决策：参数决策（Crank或平飞）
        核心任务：完成中制导
        决策内容：根据雷达照射需求，决定Crank或平飞
        """
        try:
            # 1. 获取敌机方位
            enemy_bearing = self._get_enemy_bearing(env, agent_id)
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            
            # 2. 计算航向差
            heading_diff = ((enemy_bearing - current_heading + 180) % 360) - 180
            
            # 3. 决策逻辑
            # 如果航向差<15°，平飞保持雷达照射
            # 如果航向差>15°，轻微Crank调整同时保持照射
            if abs(heading_diff) < 15.0:
                # 平飞，保持雷达照射
                if not hasattr(self, 'lr_maneuver'):
                    self.lr_maneuver = {}
                self.lr_maneuver[agent_id] = 'LEVEL_FLIGHT'
                logging.debug(f"[LR决策] {agent_id} 航向差{heading_diff:.1f}°，选择平飞")
            else:
                # 轻微Crank，调整航向同时保持照射
                if not hasattr(self, 'lr_maneuver'):
                    self.lr_maneuver = {}
                self.lr_maneuver[agent_id] = 'CRANK'
                logging.debug(f"[LR决策] {agent_id} 航向差{heading_diff:.1f}°，选择Crank")
            
        except Exception as e:
            logging.error(f"LR决策错误: {e}")
    
    def _decide_at_tr(self, env, agent_id: str):
        """
        TR节点决策：机动决策（继续进政或脱离）
        核心任务：中制导结束，规避敌方攻击
        决策内容：检查作战目标是否完成，决定继续或脱离
        """
        try:
            # 1. 检查是否已发射导弹
            has_launched = agent_id in self.missile_launched
            
            # 2. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            my_threat = threat_info.get('lead_threat', 0.5) if agent_id.endswith('100') else threat_info.get('wingman_threat', 0.5)
            
            # 3. 决策逻辑
            should_continue = True
            
            # 激进肃清：始终继续
            if self.my_intent == 'AGGRESSIVE_CLEAR':
                should_continue = True
            
            # 保守肃清：检查威胁
            elif self.my_intent == 'CONSERVATIVE_CLEAR':
                if my_threat > 0.8:
                    should_continue = False
                    logging.warning(f"⚠️ [TR决策] {agent_id} 威胁过高({my_threat:.2f})，决定脱离")
            
            # 防御意图：已发射导弹则脱离
            elif self.my_intent == 'DEFENSIVE':
                if has_launched:
                    should_continue = False
                    logging.info(f"✅ [TR决策] {agent_id} 已完成防御任务，准备脱离")
            
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
        """
        try:
            # 1. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            
            # 2. 选择规避机动
            if lead_threat > 0.7:
                evasion_maneuver = 'SHORT_SKATE'  # 高威胁：Short Skate
                logging.info(f"🔄 [DOR决策] 高威胁({lead_threat:.2f})，选择Short Skate")
            elif lead_threat > 0.4:
                evasion_maneuver = 'BEAM'  # 中威胁：Beam机动
                logging.info(f"🔄 [DOR决策] 中威胁({lead_threat:.2f})，选择Beam")
            else:
                evasion_maneuver = 'NOTCH_BACK'  # 低威胁：Notch back
                logging.info(f"🔄 [DOR决策] 低威胁({lead_threat:.2f})，选择Notch back")
            
            # 3. 预决策下一轮战术（根据当前态势）
            # 如果态势仍然有利，考虑重新进攻
            if lead_threat < 0.6:
                self.next_round_tactic = self.selected_tactic  # 继续当前战术
                logging.info(f"🎯 [DOR决策] 预决策下一轮：继续{self.selected_tactic}")
            else:
                self.next_round_tactic = None  # 不重新进攻
                logging.info(f"🎯 [DOR决策] 预决策下一轮：不重新进攻")
            
            # 记录规避机动
            self.evasion_maneuver = evasion_maneuver
            
        except Exception as e:
            logging.error(f"DOR决策错误: {e}")
    
    def _decide_at_dr(self, env):
        """
        DR节点决策：战术决策（重新进攻或返航）
        核心任务：根据态势决定是否发起下一轮进攻
        决策内容：检查敌机状态、我方状态、意图
        """
        try:
            # 1. 检查敌机状态
            enemy_alive = []
            for enemy_id in ["B0100", "B0200"]:
                if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                    enemy_alive.append(enemy_id)
            
            # 2. 获取威胁评估
            threat_info = self.threat_evaluator.evaluate_situation(env)
            lead_threat = threat_info.get('lead_threat', 0.5)
            
            # 3. 决策逻辑
            should_reengage = False
            
            # 激进肃清：只要有敌机就继续
            if self.my_intent == 'AGGRESSIVE_CLEAR':
                if len(enemy_alive) > 0:
                    should_reengage = True
                    logging.info(f"🔥 [DR决策] 激进肃清，继续进攻剩余{len(enemy_alive)}架敌机")
            
            # 保守肃清：检查威胁和敌机数量
            elif self.my_intent == 'CONSERVATIVE_CLEAR':
                if len(enemy_alive) > 0 and lead_threat < 0.6:
                    should_reengage = True
                    logging.info(f"✅ [DR决策] 保守肃清，威胁可控({lead_threat:.2f})，继续进攻")
                else:
                    logging.info(f"🚫 [DR决策] 保守肃清，威胁过高或无敌机，返航")
            
            # 防御意图：不重新进攻
            elif self.my_intent == 'DEFENSIVE':
                should_reengage = False
                logging.info(f"🛡️ [DR决策] 防御意图，不重新进攻，返航")
            
            # 记录决策结果
            self.should_reengage = should_reengage
            
            # 如果重新进攻，选择战术
            if should_reengage and hasattr(self, 'next_round_tactic') and self.next_round_tactic:
                self.selected_tactic = self.next_round_tactic
                logging.info(f"🎯 [DR决策] 重新进攻，战术: {self.selected_tactic}")
            
        except Exception as e:
            logging.error(f"DR决策错误: {e}")
    
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
                    # 获取敌机作为目标
                    if agent_id.startswith('A'):
                        target_id = "B0100" if "B0100" in env.agents and env.agents["B0100"].is_alive else "B0200"
                    else:
                        target_id = "A0100" if "A0100" in env.agents and env.agents["A0100"].is_alive else "A0200"
                    
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
        检查是否超出CAP边界（跑马道形状），如果超出则返回回到中心的指令
        
        Returns:
            (is_out_of_bounds, return_command)
            - is_out_of_bounds: bool, 是否超出边界
            - return_command: tuple or None, 如果超出边界则返回命令索引
        """
        try:
            pos = env.agents[agent_id].get_position()
            x, y, z = pos[0], pos[1], pos[2]
            
            # 设置缓冲区，提前预警
            buffer = 5000  # 5km缓冲区
            
            # 检查高度
            out_z = z < self.cap_boundary['z_min'] or z > self.cap_boundary['z_max']
            near_z = z < (self.cap_boundary['z_min'] + buffer) or z > (self.cap_boundary['z_max'] - buffer)
            
            # 检查是否在跑马道内（南北向椭圆）
            rect_x_min = self.cap_boundary['rect_x_min']
            rect_x_max = self.cap_boundary['rect_x_max']
            rect_y_min = self.cap_boundary['rect_y_min']
            rect_y_max = self.cap_boundary['rect_y_max']
            radius = self.cap_boundary['circle_radius']
            south_center = self.cap_boundary['south_circle_center']
            north_center = self.cap_boundary['north_circle_center']
            
            in_boundary = False
            
            # 检查是否在矩形部分内（中间直道）
            if rect_x_min <= x <= rect_x_max and rect_y_min <= y <= rect_y_max:
                in_boundary = True
            # 检查是否在南端半圆内
            elif x < rect_x_min:
                dist_to_south = np.sqrt((x - south_center[0])**2 + (y - south_center[1])**2)
                if dist_to_south <= radius:
                    in_boundary = True
            # 检查是否在北端半圆内
            elif x > rect_x_max:
                dist_to_north = np.sqrt((x - north_center[0])**2 + (y - north_center[1])**2)
                if dist_to_north <= radius:
                    in_boundary = True
            
            # 如果在边界内且高度正常，返回False - 允许战术自由执行
            if in_boundary and not out_z:  # 改为out_z而不是near_z，只在严格超出时拦截
                return False, None
            
            # 只有严格超出边界时才拦截战术执行
            # 计算返回目标点（返回到跑马道中心线Y=0）
            target_x = x  # 保持当前南北位置
            target_y = 0  # 返回中心线
            cap_center_z = (self.cap_boundary['z_min'] + self.cap_boundary['z_max']) / 2
            
            # 计算返回航向
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            delta_x = target_x - x
            delta_y = target_y - y
            target_bearing = np.rad2deg(np.arctan2(delta_y, delta_x))
            target_heading = (90 - target_bearing) % 360.0
            
            heading_diff = ((target_heading - current_heading + 180) % 360) - 180
            
            # 选择航向指令
            if abs(heading_diff) > 60:
                heading_cmd = 14 if heading_diff > 0 else 2
            elif abs(heading_diff) > 30:
                heading_cmd = 12 if heading_diff > 0 else 4
            elif abs(heading_diff) > 15:
                heading_cmd = 10 if heading_diff > 0 else 6
            else:
                heading_cmd = 8
            
            # 选择高度指令
            if z < self.cap_boundary['z_min']:
                alt_cmd = 10  # 爬升+300m
            elif z > self.cap_boundary['z_max']:
                alt_cmd = 4   # 下降-300m
            else:
                alt_cmd = 7   # 保持高度
            
            # 严格超出边界才返回True
            logging.warning(f"🚨 [{agent_id}] 超出CAP跑马道! "
                          f"位置:({x/1000:.1f},{y/1000:.1f},{z/1000:.0f}m) "
                          f"返回中心线 航向调整{heading_diff:+.1f}°")
            return True, (alt_cmd, heading_cmd, 5)  # 加速返回
            
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
        
        # ===== 敌方AI：完整AI系统 =====
        if agent_id.startswith('B'):
            if hasattr(self, 'enemy_ai'):
                return self.enemy_ai.get_enemy_command_indices(env, agent_id)
            else:
                # 退而求其次：平飞
                return 7, 8, 3
        
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
        
        # 战术选择日志（简化）
        
        # 4. 根据战术类型分发（7种完整战术）
        if selected_tactic == 'DRAG_SHOOT':
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
            
            logging.warning(f"[机动指令] {agent_id}")
            logging.warning(f"  机动类型: {maneuver_type}")
            logging.warning(f"  高度变化: {alt_change:+.0f}m | 航向变化: {hdg_change:+.1f}° | 速度变化: {vel_change:+.0f}m/s")
            logging.warning(f"  指令索引: alt={result[0]} hdg={result[1]} vel={result[2]}")
        
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
        
        # 选择合适的航向指令
        if abs(heading_diff) < 5.0:
            heading_cmd = 8  # 保持航向
        elif heading_diff > 0:  # 需要右转
            if abs(heading_diff) > 60:
                heading_cmd = 14  # 右转90°
            elif abs(heading_diff) > 40:
                heading_cmd = 12  # 右侧Crank +60°
            elif abs(heading_diff) > 20:
                heading_cmd = 11  # 右侧Crank +45°
            else:
                heading_cmd = 10  # 右转30°
        else:  # 需要左转
            if abs(heading_diff) > 60:
                heading_cmd = 2   # 左转90°
            elif abs(heading_diff) > 40:
                heading_cmd = 4   # 左侧Crank -60°
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
            
            # 获取敌方长机位置
            enemy_id = "B0100" if agent_id.startswith('A') else "A0100"
            if enemy_id in env.agents and env.agents[enemy_id].is_alive:
                enemy_pos = env.agents[enemy_id].get_position()
            else:
                # 敌方长机不存在，尝试僚机
                enemy_id = "B0200" if agent_id.startswith('A') else "A0200"
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
                
                # LR阶段继续保持航向，完成中制导
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
        
        # 僚机动作序列（独立阶段判断）
        else:
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
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
                
                # LR阶段保持350°航向，完成中制导
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
                
                # LR阶段继续保持航向，完成中制导
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
        
        # 僚机动作序列（后机，独立阶段判断）
        else:
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
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
                
                # LR阶段继续保持后方队形，完成中制导
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
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 僚机动作序列（独立阶段判断）
        else:
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            if wingman_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段继续保持航向，完成中制导
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
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
            target_altitude = 6096.0  # 保持低空6000m
            
            if self.current_phase in [TacticalPhase.NLT_MELD, TacticalPhase.MELD_MTR, TacticalPhase.MTR_LR]:
                # 保持低空0°航向
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif self.current_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段继续保持航向，完成中制导
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
        
        # 僚机动作序列（高空，独立阶段判断）
        else:
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            target_altitude = 8000.0  # 目标高空8000m
            
            if wingman_phase == TacticalPhase.NLT_MELD:
                # 保持10°右偏，快速建立横向分离
                return self._maintain_heading_precise(env, agent_id, 10.0)
            elif wingman_phase == TacticalPhase.MELD_MTR:
                # 爬升到高空，保持10°右偏（不是0°！）
                alt_diff = target_altitude - current_alt
                if abs(alt_diff) > 200:
                    # 需要爬升，同时保持10°右偏
                    alt_cmd = 14  # 爬升+1500m
                    hdg_cmd = self._convert_heading_to_index(np.deg2rad(10.0))
                    return alt_cmd, hdg_cmd, 3
                else:
                    # 已到达高度，保持10°右偏
                    return self._maintain_heading_precise(env, agent_id, 10.0)
            elif wingman_phase == TacticalPhase.MTR_LR:
                # 保持高空10°右偏
                return self._maintain_heading_precise(env, agent_id, 10.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段保持10°右偏，完成中制导
                return self._maintain_heading_precise(env, agent_id, 10.0)
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
            
            if lead_phase == TacticalPhase.NLT_MELD:
                # 初期保持0°，延迟展开
                if env.current_step % 60 == 0:
                    logging.info(f"     → 目标航向:0°")
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase == TacticalPhase.MELD_MTR:
                # MELD-MTR阶段：展开到左侧315°（-45°）
                if env.current_step % 60 == 0:
                    logging.info(f"     → 目标航向:315°")
                return self._maintain_heading_precise(env, agent_id, 315.0)
            elif lead_phase == TacticalPhase.MTR_LR:
                # MTR-LR阶段：收拢到0°形成钳形夹击
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段继续保持航向，完成中制导
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif lead_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 内侧返航：右转（与左侧Crank相反）
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
            else:
                return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
        
        # 僚机动作（右侧包抄，独立阶段判断）
        else:
            leader_blue = env._jsbsims.get("B0100") or env._jsbsims.get("B0200")
            if leader_blue and leader_blue.is_alive:
                distance = self._calculate_distance_between(env.agents[agent_id], leader_blue)
                wingman_phase = self._get_wingman_phase_by_distance(distance)
            else:
                wingman_phase = self.current_phase
            
            if wingman_phase == TacticalPhase.NLT_MELD:
                # 初期保持0°，延迟展开
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.MELD_MTR:
                # MELD-MTR阶段：展开到右侧45°
                return self._maintain_heading_precise(env, agent_id, 45.0)
            elif wingman_phase == TacticalPhase.MTR_LR:
                # MTR-LR阶段：收拢到0°形成钳形夹击
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.LR_TR:
                # LR阶段（78km）：发射主动雷达弹，中制导开始
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch < 0:  # 还未发射
                    self.missile_launched[agent_id] = True  # 标记需要发射
                
                # LR阶段继续保持航向，完成中制导
                return self._maintain_heading_precise(env, agent_id, 0.0)
            elif wingman_phase == TacticalPhase.TR_DOR:
                # TR阶段（75km）：中制导结束，准备规避
                last_launch = self.last_missile_launch_time.get(agent_id, -999)
                if last_launch > 0 and (current_time - last_launch) > 5.0:
                    # 内侧返航：左转（与右侧Crank相反）
                    return self._execute_short_skate_precise(env, agent_id, current_time, skate_direction)
                else:
                    return self._maintain_heading_precise(env, agent_id, 0.0)
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
        
        # TR-DOR阶段：Short Skate返航
        elif agent_phase == TacticalPhase.TR_DOR:
            direction = 'left' if is_lead else 'right'
            return self._execute_short_skate(env, agent_id, direction)
        
        # DOR-DR阶段：执行规避机动（根据DOR决策）
        elif agent_phase == TacticalPhase.DOR_DR:
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
            next_tactic = getattr(self, 'next_round_tactic', None)
            if next_tactic:
                # 重新进攻：转向敌机并加速
                enemy_bearing = self._get_enemy_bearing(env, agent_id)
                return self._turn_to_heading(env, agent_id, enemy_bearing, speed_cmd=5)
            else:
                # 不重新进攻：180°返航
                return 7, 16, 5  # 180°转向 + 大幅加速返航
        
        # BEYOND_MAR阶段：强制返航
        elif agent_phase == TacticalPhase.BEYOND_MAR:
            return 7, 16, 5  # 180°转向 + 大幅加速返航
        
        else:
            # 默认：平稳返航
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
        
        # 战术回转：180°机动偏置
        return 7, 16, 5  # 高度保持 + 180°转向 + 大幅加速
    
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
        
        # 获取战术指令索引
        altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)
        
        # 关键调试日志：每60步打印一次指令
        if env.current_step % 60 == 0 and agent_id.startswith('A'):
            alt_change = self.norm_delta_altitude[altitude_cmd_id] * 1000
            hdg_change = np.rad2deg(self.norm_delta_heading[heading_cmd_id])
            vel_change = self.norm_delta_velocity[velocity_cmd_id] * 100
            current_heading = env.agents[agent_id].get_property_value(c.attitude_psi_deg)
            phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            logging.info(f"🔧 [{agent_id}] 阶段:{phase.value} 当前航向:{current_heading:.1f}° 指令:alt={altitude_cmd_id},hdg={heading_cmd_id}({hdg_change:+.1f}°),vel={velocity_cmd_id}")
        
        # 敌方导弹发射检查（在LR_TR阶段）
        if agent_id.startswith('B') and hasattr(self, 'enemy_ai'):
            phase = self.agent_phases.get(agent_id, TacticalPhase.NLT_MELD)
            if phase == TacticalPhase.LR_TR:
                # 在LR阶段（78km）敌方也发射导弹
                current_time = env.current_step * env.time_interval
                self.enemy_ai.handle_missile_launch(env, agent_id, current_time)
        
        # 使用baseline模型生成底层控制
        return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)
    
    def _execute_beam_maneuver(self, env, agent_id: str) -> tuple:
        """
        执行Beam机动 - 三九机动，将敌机置于自身3/9位置
        目标：通过转向使敌机位于相对方位角的90°位置（左或右侧）
        """
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
        
        logging.info(f"📍 [BEAM] {agent_id} 执行三九机动: 敌机方位{enemy_bearing:.1f}° → {beam_direction} 目标{target_heading:.1f}°")
        
        # 转向目标航向
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
