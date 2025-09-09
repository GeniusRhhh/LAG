#!/usr/bin/env python3
"""
钳形夹击战术任务 - 完全基于drag_shoot_tactical_task架构模式
严格照抄drag_shoot_tactical_task的实现，只修改战术逻辑部分
"""

import logging
import numpy as np
import torch
from enum import Enum
from envs.JSBSim.tasks.multiplecombat_task import MultipleCombatTask
from envs.JSBSim.model.baseline_actor import BaselineActor
from envs.JSBSim.utils.utils import get_root_dir
from envs.JSBSim.core.catalog import Catalog as c
from envs.JSBSim.termination_conditions.termination_condition_base import BaseTerminationCondition
from envs.JSBSim.tasks.pure_maneuvers import BasicManeuvers, CompositeManeuverExecutor


class PincerAttackTermination(BaseTerminationCondition):
    """钳形夹击专用终止条件 - 只有双方全灭才终止"""

    def __init__(self, config):
        super().__init__(config)
        self.altitude_limit = getattr(config, 'altitude_limit', 1000)  # 1000米
        self.max_steps = getattr(config, 'max_steps', 2500)

    def get_termination(self, task, env, agent_id, info={}):
        """
        自定义终止条件：
        1. 高度过低
        2. 时间限制
        3. 极端状态
        4. 过载
        5. 双方全灭（不是单架飞机被击落）
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= self.altitude_limit:
            self.log(f"{agent_id} altitude too low: {current_alt:.1f}m")
            return True, False, info

        # 检查时间限制
        if env.current_step >= self.max_steps:
            self.log(f"Time limit reached: {env.current_step} steps")
            return True, False, info

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            self.log(f"{agent_id} extreme state detected")
            return True, False, info

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            self.log(f"{agent_id} overload detected")
            return True, False, info

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            self.log("Red team eliminated")
            return True, True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            self.log("Blue team eliminated")
            return True, True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, False, info


class TacticalPhase(Enum):
    """钳形夹击战术阶段 - 基于控制距离的精确定义"""
    NLT_MELD = "NLT_MELD"    # 90-81km: 钳形展开阶段
    MELD_MTR = "MELD_MTR"    # 81-45km: 钳形收拢阶段
    MTR_TR = "MTR_TR"        # 45-41km: 导弹发射阶段
    TR_DOR = "TR_DOR"        # 41-19.6km: 内侧脱离阶段
    DOR_DR = "DOR_DR"        # 19.6-14.5km: 返航阶段


class PincerAttackTacticalTask(MultipleCombatTask):
    """
    钳形夹击战术任务 - 基于drag_shoot_tactical_task架构模式

    动作空间架构：
    - 定义的动作空间: [41, 41, 41, 30] (继承自MultipleCombatTask，但实际不使用)
    - 实际使用的动作空间: [15, 17, 7] (高层战术指令)
    - 转换机制: 高层指令 → baseline模型 → 底层飞行控制

    主要函数调用链：
    normalize_action() → _process_pincer_attack_tactics() → _get_tactical_command_indices()
    → _get_friendly_command_indices() / _get_enemy_command_indices()
    → _use_lowlevel_policy() → baseline模型输出底层控制指令

    平稳飞行指令: [7, 8, 3] = [高度0m变化, 航向0°变化, 速度0m/s变化]
    """

    def __init__(self, config):
        """初始化钳形夹击战术任务"""
        super().__init__(config)

        # 钳形夹击战术距离配置 - 严格按照控制距离定义
        self.tactical_distances = {
            'NLT_MELD_min': 81000,   # 81km - 钳形展开阶段
            'MELD_MTR_min': 45000,   # 45km - 钳形收拢阶段
            'MTR_TR_min': 41000,     # 41km - 导弹发射阶段
            'TR_DOR_min': 19600,     # 19.6km - 内侧脱离阶段
            'DOR_DR_min': 14500,     # 14.5km - 返航阶段
        }

        # 钳形夹击战术参数 - 增大crank角度形成更明显的钳形包围态势
        self.pincer_config = {
            'crank_angle_moderate': 45.0,   # 增大的Crank角度，形成更明显的钳形态势
            'leader_left_crank': True,      # 长机执行左侧Crank
            'wingman_right_crank': True,    # 僚机执行右侧Crank
            'short_skate_angle': 45.0,      # Short Skate转弯角度
            'rtb_heading': 180.0,           # 返航航向
        }

        # 当前战术阶段
        self.current_phase = TacticalPhase.NLT_MELD
        self.phase_start_time = 0.0
        self.phase_start_step = 0

        # 底层策略网络 - 完全照抄drag_shoot_tactical_task
        self.my_lowlevel_policy = BaselineActor()
        self._inner_rnn_states = {}
        self._load_baseline_models()

        # 动作空间定义：[15, 17, 7] - 高层战术指令空间
        self.norm_delta_altitude = np.array([
            -1500, -1000, -750, -500, -300, -150, -50, 0, 50, 150, 300, 500, 750, 1000, 1500
        ]) / 1000.0  # 索引7 = 0m变化（平稳飞行）

        self.norm_delta_heading = np.array([
            -60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60
        ]) * np.pi / 180.0  # 索引8 = 0°变化（保持航向）

        self.norm_delta_velocity = np.array([
            -100, -50, -20, 0, 20, 50, 100
        ])  # 索引3 = 0m/s变化（保持速度）

        # 基础机动系统 - 照抄drag_shoot_tactical_task
        self.basic_maneuvers = BasicManeuvers()
        self.composite_executor = CompositeManeuverExecutor()

        # 统一雷达管理器 - 支持多项目复用
        try:
            from radar_manager import get_unified_radar_manager
            self.radar_manager = get_unified_radar_manager()
            logging.info("📡 统一雷达管理系统已集成到钳形夹击任务")
        except ImportError as e:
            logging.warning(f"❌ 统一雷达管理器导入失败: {e}")
            self.radar_manager = None

        # 统一敌方战术AI系统 - 支持多项目复用
        try:
            from unified_enemy_tactical_ai import UnifiedEnemyTacticalAI
            self.unified_enemy_ai = UnifiedEnemyTacticalAI()
            logging.info("🤖 统一敌方战术AI系统已集成到钳形夹击任务")
        except ImportError as e:
            logging.warning(f"❌ 统一敌方战术AI系统导入失败: {e}")
            self.unified_enemy_ai = None

        # 状态跟踪
        self.initial_heading = {}
        self.initial_altitude = {}
        self.active_maneuvers = {}
        self.maneuver_start_times = {}

        # 导弹发射管理
        self.last_missile_launch_time = {}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却

        # Short Skate机动状态跟踪 - 复制拖曳射击项目
        self.short_skate_states = {}

        # 导弹发射管理 - 复制拖曳射击项目
        self.last_missile_launch_time = {"A0100": -999, "A0200": -999, "B0100": -999, "B0200": -999}
        self.friendly_missile_cooldown = 2.0  # 友方2秒冷却
        self.enemy_missile_cooldown = 10.0    # 敌方10秒冷却

        # ===== 敌方对抗机动参数 =====
        # 角色分配：B0100为shooter，B0200为support
        self.enemy_roles = {"B0100": "shooter", "B0200": "support"}
        
        # 敌方对抗阶段状态跟踪
        self.enemy_combat_states = {}
        
        # 敌方对抗距离阈值（与我方战术阶段对应）
        self.enemy_combat_thresholds = {
            'OBSERVATION': 70000,    # 70km - 远距离观察阶段
            'ENGAGEMENT': 50000,     # 50km - 中距离对抗阶段  
            'ATTACK': 40000,         # 40km - 近距离攻击阶段
            'DISENGAGE': 25000,      # 25km - 脱离阶段
            'RTB': 15000,            # 15km - 返航阶段
        }
        
        # 敌方机动时间参数
        self.enemy_maneuver_times = {
            'crank_duration': 25.0,      # Crank机动持续时间
            'attack_duration': 30.0,     # 攻击阶段持续时间
            'disengage_duration': 20.0,  # 脱离阶段持续时间
            'rtb_duration': 40.0,        # 返航阶段持续时间
        }

        logging.info("PincerAttackTacticalTask initialized")

    def get_termination(self, env, agent_id, info={}):
        """
        完全重写终止条件 - 绕过父类的termination_conditions列表
        只有双方全灭、高度过低、或达到时间限制才终止
        """
        # 检查高度过低
        current_alt = env.agents[agent_id].get_property_value(c.position_h_sl_m)
        if current_alt <= 1000:  # 1000米以下
            logging.info(f"{agent_id} altitude too low: {current_alt:.1f}m - terminating")
            return True, {"termination_reason": "low_altitude"}

        # 检查时间限制
        if env.current_step >= 2500:  # 最大步数
            logging.info(f"Time limit reached: {env.current_step} steps - terminating")
            return True, {"termination_reason": "timeout"}

        # 检查极端状态
        if env.agents[agent_id].get_property_value(c.detect_extreme_state):
            logging.info(f"{agent_id} extreme state detected - terminating")
            return True, {"termination_reason": "extreme_state"}

        # 检查过载状态
        if (abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_x_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_y_norm)) > 15.0 or
            abs(env.agents[agent_id].get_property_value(c.accelerations_n_pilot_z_norm) + 1) > 15.0):
            logging.info(f"{agent_id} overload detected - terminating")
            return True, {"termination_reason": "overload"}

        # 检查双方存活情况 - 只有当一方全灭时才终止
        red_alive = [aid for aid in ["A0100", "A0200"] if aid in env.agents and env.agents[aid].is_alive]
        blue_alive = [aid for aid in ["B0100", "B0200"] if aid in env.agents and env.agents[aid].is_alive]

        # 添加调试日志
        if env.current_step % 50 == 0:  # 每50步输出一次状态
            logging.info(f"TERMINATION CHECK - Step {env.current_step}: Red alive: {red_alive}, Blue alive: {blue_alive}")

        # 只有当一方全灭时才终止
        if len(red_alive) == 0:
            logging.info("Red team eliminated - terminating")
            return True, {"termination_reason": "red_eliminated", "winner": "blue"}
        elif len(blue_alive) == 0:
            logging.info("Blue team eliminated - terminating")
            return True, {"termination_reason": "blue_eliminated", "winner": "red"}

        # 继续仿真 - 不因为单架飞机被击落而终止
        return False, info

    def _load_baseline_models(self):
        """加载baseline模型 - 完全学习drag_shoot_tactical_task的实现"""
        try:
            model_path = get_root_dir() + '/model/baseline_model.pt'
            if torch.cuda.is_available():
                device = torch.device("cuda")
                checkpoint = torch.load(model_path, weights_only=True)
            else:
                device = torch.device("cpu")
                checkpoint = torch.load(model_path, map_location=device, weights_only=True)
            self.my_lowlevel_policy.load_state_dict(checkpoint)
            self.my_lowlevel_policy.eval()
            logging.info("Successfully loaded baseline model for pincer-attack task")
        except Exception as e:
            logging.error(f"加载baseline模型失败: {e}")
            self.my_lowlevel_policy = None

    def normalize_action(self, env, agent_id, action):
        """
        钳形夹击战术的normalize_action实现 - 完全照抄drag_shoot_tactical_task的架构
        """
        if agent_id not in env.agents or not env.agents[agent_id].is_alive:
            return np.array([0.0, 0.0, 0.0, 0.7])

        current_time = env.current_step * env.time_interval

        # 处理钳形夹击战术逻辑
        result = self._process_pincer_attack_tactics(env, agent_id, current_time)

        # 处理导弹发射 - 添加友方导弹发射功能
        self._handle_missile_launch(env, agent_id, current_time)

        return result

    def _process_pincer_attack_tactics(self, env, agent_id, current_time):
        """处理钳形夹击战术逻辑 - 照抄drag_shoot_tactical_task的_process_drag_shoot_tactics"""
        try:
            # 更新战术阶段
            self._update_tactical_phase(env)

            # 获取战术指令索引
            altitude_cmd_id, heading_cmd_id, velocity_cmd_id = self._get_tactical_command_indices(env, agent_id)

            # 使用底层策略
            return self._use_lowlevel_policy(env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id)

        except Exception as e:
            logging.error(f"钳形夹击战术逻辑失败 {agent_id}: {e}")
            return self._use_lowlevel_policy(env, agent_id, 7, 8, 3)  # 平稳飞行

    def _update_tactical_phase(self, env):
        """更新战术阶段 - 照抄drag_shoot_tactical_task的实现"""
        # 获取主要对抗双方
        leader_red = env._jsbsims.get("A0100")
        leader_blue = env._jsbsims.get("B0100")

        if not leader_red or not leader_blue or not leader_red.is_alive or not leader_blue.is_alive:
            return

        # 计算距离
        distance = self._calculate_distance(leader_red, leader_blue)

        # 确定当前阶段
        new_phase = self._get_phase_by_distance(distance)

        if new_phase != self.current_phase:
            current_time = env.current_step * env.time_interval
            logging.info(f"Phase transition: {self.current_phase.value} -> {new_phase.value} "
                        f"at t={current_time:.1f}s, distance={distance/1000:.1f}km")
            self.current_phase = new_phase
            self.phase_start_time = current_time
            self.phase_start_step = env.current_step

    def _get_phase_by_distance(self, distance):
        """根据距离确定战术阶段"""
        if distance >= self.tactical_distances['NLT_MELD_min']:
            return TacticalPhase.NLT_MELD
        elif distance >= self.tactical_distances['MELD_MTR_min']:
            return TacticalPhase.MELD_MTR
        elif distance >= self.tactical_distances['MTR_TR_min']:
            return TacticalPhase.MTR_TR
        elif distance >= self.tactical_distances['TR_DOR_min']:
            return TacticalPhase.TR_DOR
        else:
            return TacticalPhase.DOR_DR

    def _calculate_distance(self, aircraft1, aircraft2):
        """计算两架飞机之间的距离"""
        pos1 = aircraft1.get_position()
        pos2 = aircraft2.get_position()
        return np.linalg.norm(pos1 - pos2)

    def _get_tactical_command_indices(self, env, agent_id):
        """获取钳形夹击战术指令索引"""
        # 根据当前阶段和角色确定战术行为
        if agent_id.startswith('A'):  # 友方
            return self._get_friendly_command_indices(env, agent_id)
        else:  # 敌方
            return self._get_enemy_command_indices(env, agent_id)

    def _get_friendly_command_indices(self, env, agent_id):
        """友方钳形夹击指令 - 严格按照战术规格实现"""
        if agent_id == "A0100":  # 友方长机
            return self._get_friendly_leader_command_indices(env, agent_id)
        elif agent_id == "A0200":  # 友方僚机
            return self._get_friendly_wingman_command_indices(env, agent_id)
        else:
            return 7, 8, 3  # 默认平稳飞行

    def _get_friendly_leader_command_indices(self, env, agent_id):
        """友方长机钳形夹击指令"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1 (90-81km): 长机执行LEFT crank机动 - 增大角度形成更明显的钳形态势
            return self._maintain_heading_precise(env, agent_id, 315.0)  # 左转45°

        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2 (81-45km): Crank恢复，指向敌机方向，平行飞行0°
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3 (45-41km): 导弹发射阶段，保持0°航向
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4 (41-19.6km): 长机执行RIGHT转弯进行Short Skate
            return self._execute_short_skate_precise(env, agent_id, "right")

        else:  # DOR_DR
            # 阶段5 (19.6-14.5km): 返航，航向180°
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _get_friendly_wingman_command_indices(self, env, agent_id):
        """友方僚机钳形夹击指令"""
        if self.current_phase == TacticalPhase.NLT_MELD:
            # 阶段1 (90-81km): 僚机执行RIGHT crank机动 - 增大角度形成更明显的钳形态势
            return self._maintain_heading_precise(env, agent_id, 45.0)  # 右转45°

        elif self.current_phase == TacticalPhase.MELD_MTR:
            # 阶段2 (81-45km): Crank恢复，指向敌机方向，平行飞行0°
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.MTR_TR:
            # 阶段3 (45-41km): 导弹发射阶段，僚机可能稍微滞后
            return self._maintain_heading_precise(env, agent_id, 0.0)

        elif self.current_phase == TacticalPhase.TR_DOR:
            # 阶段4 (41-19.6km): 僚机执行LEFT转弯进行Short Skate
            return self._execute_short_skate_precise(env, agent_id, "left")

        else:  # DOR_DR
            # 阶段5 (19.6-14.5km): 返航，航向180°
            return self._maintain_heading_precise(env, agent_id, 180.0)

    def _get_enemy_command_indices(self, env, agent_id):
        """敌方战术指令索引 - 使用统一敌方AI系统"""
        current_time = env.current_step * env.time_interval

        try:
            # 检查是否有集成的统一敌方AI系统
            if hasattr(self, 'unified_enemy_ai'):
                altitude_cmd, heading_cmd, velocity_cmd = self.unified_enemy_ai.get_enemy_action(
                    env, agent_id, current_time
                )
                logging.debug(f"✅ {agent_id} 统一敌方AI指令: ({altitude_cmd}, {heading_cmd}, {velocity_cmd})")
                return altitude_cmd, heading_cmd, velocity_cmd
            else:
                logging.warning(f"⚠️ {agent_id} 未找到统一敌方AI系统，使用备用方案")
                return self._get_enemy_command_indices_fallback(env, agent_id, current_time)

        except Exception as e:
            logging.error(f"统一敌方AI执行错误: {e}")
            return self._get_enemy_command_indices_fallback(env, agent_id, current_time)

    def _get_enemy_command_indices_fallback(self, env, agent_id, current_time):
        """敌方对抗机动逻辑：真正模仿我方的控制距离时间线
        与我方钳形战术形成真实对抗，每个阶段都有明确的航向控制"""
        
        # 确保敌方角色分配
        self._ensure_enemy_roles(env)
        
        # 获取当前态势
        my_ac = env.agents[agent_id]
        my_pos = my_ac.get_position()
        my_heading = np.rad2deg(my_ac.get_property_value(c.attitude_psi_rad))
        my_alt = my_pos[2]
        
        # 计算到最近我方目标的距离和方位
        closest_dist = float('inf')
        closest_bearing = 0.0
        closest_target = None
        
        for fid in ["A0100", "A0200"]:
            if fid in env.agents and env.agents[fid].is_alive:
                tgt_pos = env.agents[fid].get_position()
                d = np.linalg.norm(tgt_pos - my_pos)
                if d < closest_dist:
                    closest_dist = d
                    closest_bearing = self._calculate_bearing_to_target(my_pos, tgt_pos)
                    closest_target = fid
        
        # 无目标时返航
        if closest_dist == float('inf'):
            return self._maintain_heading_precise(env, agent_id, 0.0)
        
        # 初始化敌方对抗状态
        if agent_id not in self.enemy_combat_states:
            # 固定侧向：shooter偏左(-1)，support偏右(+1)
            role = self.enemy_roles.get(agent_id, 'support')
            side_sign = -1 if role == 'shooter' else 1
            self.enemy_combat_states[agent_id] = {
                'phase': 'OBSERVATION',
                'phase_start_time': current_time,
                'initial_heading': my_heading,
                'initial_altitude': my_alt,
                'last_shot_time': -999.0,
                'side_sign': side_sign,
                'hold_heading': None,
                'hold_until': 0.0,
                'has_fired': False
            }
        
        state = self.enemy_combat_states[agent_id]
        role = self.enemy_roles.get(agent_id, 'support')
        side_sign = state.get('side_sign', -1 if role == 'shooter' else 1)
        
        # 若存在保持机动（notch/cold），在时间窗口内维持
        if state.get('hold_heading') is not None and current_time < state.get('hold_until', 0.0):
            return self._maintain_heading_precise(env, agent_id, state['hold_heading'])

        # 检查来袭导弹威胁：触发一次notch并保持数秒，避免抖动
        if self._check_missile_threat(env, agent_id):
            # 动态notch保持：距离近保持更短，并加入轻微下降（通过低速俯仰由低层控制处理，这里仅做航向）
            notch_heading = (closest_bearing + side_sign * 90.0) % 360.0
            # 基础距离近似：用closest_dist估计，<30km -> 2s，30–60km -> 4s，>60km -> 6s
            if closest_dist < 30000:
                hold_time = 2.0
            elif closest_dist < 60000:
                hold_time = 4.0
            else:
                hold_time = 6.0
            state['hold_heading'] = notch_heading
            state['hold_until'] = current_time + hold_time
            logging.debug(f"敌方{agent_id}执行notch规避，转向{notch_heading:.1f}°，保持{hold_time:.1f}s 至{state['hold_until']:.1f}s")
            return self._maintain_heading_precise(env, agent_id, notch_heading)
        
        # 根据距离确定对抗阶段（与我方完全对应）
        current_phase = self._determine_enemy_phase(closest_dist)
        
        # 阶段切换处理
        if state['phase'] != current_phase:
            state['phase'] = current_phase
            state['phase_start_time'] = current_time
            logging.info(f"敌方{agent_id}进入{current_phase}阶段，距离{closest_dist/1000:.1f}km")
        
        # 执行对应阶段的机动（真正模仿我方的逻辑）
        if current_phase == 'OBSERVATION':
            return self._execute_observation_phase(env, agent_id, current_time, state, closest_bearing, role)
        elif current_phase == 'ENGAGEMENT':
            return self._execute_engagement_phase(env, agent_id, current_time, state, closest_bearing, role)
        elif current_phase == 'ATTACK':
            return self._execute_attack_phase(env, agent_id, current_time, state, closest_bearing, role)
        elif current_phase == 'DISENGAGE':
            return self._execute_disengage_phase(env, agent_id, current_time, state, closest_bearing, role)
        else:  # RTB阶段
            return self._execute_rtb_phase(env, agent_id, current_time, state)
    
    def _determine_enemy_phase(self, distance):
        """根据距离确定敌方对抗阶段（与我方战术阶段完全对应）"""
        if distance >= 81000:  # 81km - 对应我方NLT_MELD
            return 'OBSERVATION'
        elif distance >= 45000:  # 45km - 对应我方MELD_MTR
            return 'ENGAGEMENT'
        elif distance >= 41000:  # 41km - 对应我方MTR_TR
            return 'ATTACK'
        elif distance >= 19600:  # 19.6km - 对应我方TR_DOR
            return 'DISENGAGE'
        else:
            return 'RTB'
    
    def _check_missile_threat(self, env, agent_id):
        """检查是否有来袭导弹威胁"""
        if hasattr(env, '_missile_records'):
            for mid, m in getattr(env, '_missile_records', {}).items():
                if m.get('status') == 'LAUNCHED' and m.get('target') == agent_id:
                    return True
        return False

    # 旧的notch函数已删除，使用基础的notch规避逻辑
    
    def _execute_observation_phase(self, env, agent_id, current_time, state, closest_bearing, role):
        """远距离观察阶段：朝向目标为主，混合初始航向，避免早期交叉和过度偏向"""
        side_sign = state.get('side_sign', -1 if role == 'shooter' else 1)
        # 混合初始航向与目标方位：初期占比更多初始航向，逐步向目标方位过渡
        t = max(0.0, min((current_time - state['phase_start_time']) / 12.0, 1.0))  # 12秒内线性过渡
        base_heading = (1 - t) * state['initial_heading'] + t * closest_bearing
        target_heading = (base_heading + side_sign * 5.0) % 360.0
        logging.debug(f"敌方{agent_id}观察阶段：blend={t:.2f}, 初始{state['initial_heading']:.1f}°, 目标{closest_bearing:.1f}°, 输出{target_heading:.1f}°")
        return self._maintain_heading_precise(env, agent_id, target_heading)

    def _execute_engagement_phase(self, env, agent_id, current_time, state, closest_bearing, role):
        """中距离对抗阶段：固定侧向的crank（±30°），但相对目标方位进行限制，避免偏到不对称方向"""
        side_sign = state.get('side_sign', -1 if role == 'shooter' else 1)
        # 相对目标进行±30°，但限制与初始航向的偏差，避免交叉过大
        desired = (closest_bearing + side_sign * 30.0) % 360.0
        # 用最小角差向desired靠近，限制每步最大转向（平滑）
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        turn_limit = 15.0  # 每阶段调用限制单步转向
        diff = ((desired - current_heading + 540) % 360) - 180
        target_heading = (current_heading + np.clip(diff, -turn_limit, turn_limit)) % 360.0
        logging.debug(f"敌方{agent_id}对抗阶段：desired={desired:.1f}°, 当前{current_heading:.1f}°, 输出{target_heading:.1f}°")
        return self._maintain_heading_precise(env, agent_id, target_heading)
    
    def _execute_attack_phase(self, env, agent_id, current_time, state, closest_bearing, role):
        """近距离攻击阶段：收紧至±15–20°，shooter优先保持攻角；若刚发射则触发短cold保持"""
        side_sign = state.get('side_sign', -1 if role == 'shooter' else 1)
        desired = (closest_bearing + side_sign * 18.0) % 360.0
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))
        turn_limit = 20.0
        diff = ((desired - current_heading + 540) % 360) - 180
        target_heading = (current_heading + np.clip(diff, -turn_limit, turn_limit)) % 360.0
        logging.debug(f"敌方{agent_id}攻击阶段：desired={desired:.1f}°, 当前{current_heading:.1f}°, 输出{target_heading:.1f}°")
        return self._maintain_heading_precise(env, agent_id, target_heading)
    
    def _execute_disengage_phase(self, env, agent_id, current_time, state, closest_bearing, role):
        """脱离阶段：执行冷转向（cold）保持若干秒，然后转入RTB门槛逻辑"""
        # 根据固定侧向执行冷转向：背离目标30–60°，并保持数秒避免来回摇摆
        side_sign = state.get('side_sign', -1 if role == 'shooter' else 1)
        desired = (closest_bearing + side_sign * 70.0) % 360.0
        state['hold_heading'] = desired
        state['hold_until'] = current_time + 8.0
        logging.debug(f"敌方{agent_id}脱离阶段：cold至{desired:.1f}°，保持至{state['hold_until']:.1f}s")
        return self._maintain_heading_precise(env, agent_id, desired)

    def _execute_rtb_phase(self, env, agent_id, current_time, state):
        """返航阶段（对应我方DOR_DR）：稳定返航"""
        # 与我方相反：我方返航180°时，敌方返航0°
        target_heading = 0.0  # 北向返航
        
        logging.debug(f"敌方{agent_id}返航阶段：转向{target_heading:.1f}°")
        return self._maintain_heading_precise(env, agent_id, target_heading)

    # 旧的FSM函数已删除，使用新的对抗阶段逻辑

    def _ensure_enemy_roles(self, env):
        """确保始终有一个shooter；若当前shooter阵亡则将support提升为shooter。"""
        # 先确认现有分配是否有效
        shooter_alive = False
        for eid, role in list(self.enemy_roles.items()):
            if role == 'shooter' and eid in env.agents and env.agents[eid].is_alive:
                shooter_alive = True
                break
        if shooter_alive:
            return
        # shooter阵亡则寻找一个support接替
        for eid in ['B0100', 'B0200']:
            if eid in env.agents and env.agents[eid].is_alive:
                # 设置该机为shooter，其余为support
                for k in list(self.enemy_roles.keys()):
                    self.enemy_roles[k] = 'support'
                self.enemy_roles[eid] = 'shooter'
                return

    def _assess_enemy_threats(self, env, agent_id, current_time):
        """评估敌方威胁等级"""
        enemy_aircraft = env.agents[agent_id]
        enemy_pos = enemy_aircraft.get_position()
        
        # 检查导弹威胁
        missile_threat = False
        missile_distance = float('inf')
        
        # 检查环境中的导弹
        if hasattr(env, '_missile_records'):
            for missile_id, missile_data in env._missile_records.items():
                if missile_data['status'] == 'LAUNCHED' and missile_data['target'] == agent_id:
                    # 计算导弹距离和威胁
                    missile_threat = True
                    missile_distance = self._calculate_distance(enemy_aircraft, env.agents[missile_data['launcher']])
                    break
        
        # 检查友方飞机威胁
        friendly_threats = []
        for friendly_id in ["A0100", "A0200"]:
            if friendly_id in env.agents and env.agents[friendly_id].is_alive:
                distance = self._calculate_distance(enemy_aircraft, env.agents[friendly_id])
                friendly_threats.append({
                    'id': friendly_id,
                    'distance': distance,
                    'bearing': self._calculate_bearing_to_target(enemy_pos, env.agents[friendly_id].get_position())
                })
        
        # 威胁等级评估
        threat_level = 'low'
        if missile_threat and missile_distance < 30000:  # 30km内有导弹威胁
            threat_level = 'critical'
        elif missile_threat:
            threat_level = 'high'
        elif friendly_threats and min([t['distance'] for t in friendly_threats]) < 25000:  # 25km内有敌机
            threat_level = 'high'
        elif friendly_threats and min([t['distance'] for t in friendly_threats]) < 40000:  # 40km内有敌机
            threat_level = 'medium'
        
        # 攻击机会评估
        attack_opportunity = False
        if friendly_threats:
            closest_friendly = min(friendly_threats, key=lambda x: x['distance'])
            if closest_friendly['distance'] <= 50000:  # 50km内有攻击机会
                attack_opportunity = True
        
        return {
            'missile_threat': missile_threat,
            'missile_distance': missile_distance,
            'threat_level': threat_level,
            'friendly_threats': friendly_threats,
            'attack_opportunity': attack_opportunity,
            'current_time': current_time
        }

    def _execute_evasive_maneuver(self, env, agent_id, current_time, threat_assessment):
        """执行规避机动"""
        enemy_aircraft = env.agents[agent_id]
        current_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))
        current_altitude = self._safe_get_altitude(enemy_aircraft)
        
        # 根据威胁方向选择规避策略
        if threat_assessment['friendly_threats']:
            closest_threat = min(threat_assessment['friendly_threats'], key=lambda x: x['distance'])
            threat_bearing = closest_threat['bearing']
            
            # 计算规避航向（远离威胁）
            evasive_heading = (threat_bearing + 180) % 360
            
            # 高度变化：随机上下机动
            altitude_change = np.random.choice([-500, 500, -1000, 1000])
            target_altitude = current_altitude + altitude_change
            
            # 执行规避机动
            print(f"🚶 {agent_id}: 执行规避机动 - 威胁距离{closest_threat['distance']/1000:.1f}km, 规避航向{evasive_heading:.1f}°")
            
            # 转换为动作索引
            alt_cmd = self._convert_altitude_to_index(target_altitude - current_altitude)
            hdg_cmd = self._convert_heading_to_index(np.deg2rad(evasive_heading - current_heading))
            vel_cmd = 3  # 保持当前速度
            
            return alt_cmd, hdg_cmd, vel_cmd
        
        # 默认规避动作
        return 7, 8, 3

    def _execute_defensive_tactics(self, env, agent_id, current_time, threat_assessment):
        """执行防御战术"""
        enemy_aircraft = env.agents[agent_id]
        current_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))

        # 执行Notch机动（90度侧向规避）
        if threat_assessment['friendly_threats']:
            closest_threat = min(threat_assessment['friendly_threats'], key=lambda x: x['distance'])
            threat_bearing = closest_threat['bearing']
            
            # Notch机动：侧向90度
            notch_heading = (threat_bearing + 90) % 360
            if np.random.random() < 0.5:
                notch_heading = (threat_bearing - 90) % 360
            
            print(f"🛡️ {agent_id}: 执行Notch防御机动 - 威胁方向{threat_bearing:.1f}°, Notch航向{notch_heading:.1f}°")
            
            # 转换为动作索引
            hdg_cmd = self._convert_heading_to_index(np.deg2rad(notch_heading - current_heading))
            return 7, hdg_cmd, 3
        
        return 7, 8, 3

    def _execute_attack_tactics(self, env, agent_id, current_time, threat_assessment):
        """执行攻击战术"""
        enemy_aircraft = env.agents[agent_id]
        current_heading = np.rad2deg(enemy_aircraft.get_property_value(c.attitude_psi_rad))
        
        if threat_assessment['friendly_threats']:
            closest_target = min(threat_assessment['friendly_threats'], key=lambda x: x['distance'])
            target_bearing = closest_target['bearing']
            
            # 攻击机动：Crank + 高度变化
            crank_angle = np.random.choice([30, -30, 45, -45])
            attack_heading = (target_bearing + crank_angle) % 360
            
            # 高度变化：随机上下
            altitude_change = np.random.choice([-300, 300, -600, 600])
            current_altitude = self._safe_get_altitude(enemy_aircraft)
            target_altitude = current_altitude + altitude_change
            
            print(f"⚔️ {agent_id}: 执行攻击机动 - 目标距离{closest_target['distance']/1000:.1f}km, 攻击航向{attack_heading:.1f}°")
            
            # 转换为动作索引
            alt_cmd = self._convert_altitude_to_index(target_altitude - current_altitude)
            hdg_cmd = self._convert_heading_to_index(np.deg2rad(attack_heading - current_heading))
            vel_cmd = 3  # 保持当前速度
            
            return alt_cmd, hdg_cmd, vel_cmd
        
        return 7, 8, 3

    def _execute_standard_bvr(self, env, agent_id, current_time, threat_assessment):
        """基础默认：指向最近友机航向，避免高级逻辑"""
        my_ac = env.agents[agent_id]
        my_pos = my_ac.get_position()
        closest_dist = float('inf')
        closest_bearing = 0.0
        for fid in ["A0100", "A0200"]:
            if fid in env.agents and env.agents[fid].is_alive:
                tgt_pos = env.agents[fid].get_position()
                d = np.linalg.norm(tgt_pos - my_pos)
                if d < closest_dist:
                    closest_dist = d
                    closest_bearing = self._calculate_bearing_to_target(my_pos, tgt_pos)
        if closest_dist == float('inf'):
            return 7, 8, 3
        return self._maintain_heading_precise(env, agent_id, closest_bearing)

    def _convert_altitude_to_index(self, altitude_change_m):
        """将高度变化转换为动作索引"""
        altitude_change_km = altitude_change_m / 1000.0
        
        # 找到最接近的高度变化索引
        differences = np.abs(self.norm_delta_altitude - altitude_change_km)
        return np.argmin(differences)

    def _safe_get_altitude(self, agent):
        """安全获取飞机高度"""
        try:
            position = agent.get_position()
            if hasattr(position, '__len__') and len(position) >= 3:
                return float(position[2])
        except:
            pass
        
        try:
            from envs.JSBSim.core.catalog import ExtraCatalog
            return agent.get_property_value(ExtraCatalog.position_h_sl_m)
        except:
            pass
        
        return 6000.0  # 默认高度

    # ========== 精确航向保持和机动方法 ==========
    def _maintain_heading_precise(self, env, agent_id, target_heading):
        """精确航向保持 - 复制拖曳射击项目的实现"""
        current_heading = np.rad2deg(env.agents[agent_id].get_property_value(c.attitude_psi_rad))

        # 计算航向差值
        heading_diff = target_heading - current_heading
        while heading_diff > 180:
            heading_diff -= 360
        while heading_diff < -180:
            heading_diff += 360

        # 精确航向控制
        if abs(heading_diff) > 2.0:
            heading_cmd_id = self._convert_heading_to_index(np.deg2rad(heading_diff))
        else:
            heading_cmd_id = 8  # 保持当前航向

        print(f"[钳形夹击] {agent_id}: 当前航向{current_heading:.1f}° -> 目标航向{target_heading:.1f}° (差值{heading_diff:.1f}°)")

        return 7, heading_cmd_id, 3  # 保持高度，调整航向，保持速度

    def _execute_short_skate_precise(self, env, agent_id, direction):
        """执行精确的Short Skate机动 - 复制拖曳射击项目的实现"""
        if direction == "right":
            # 长机右转Short Skate
            target_heading = 45.0  # 右转45°然后逐渐转向180°
        elif direction == "left":
            # 僚机左转Short Skate
            target_heading = 315.0  # 左转45°然后逐渐转向180°
        else:  # direction == "north" for enemy
            # 敌方返航北方
            target_heading = 0.0

        # 初始化Short Skate状态
        if agent_id not in self.short_skate_states:
            self.short_skate_states[agent_id] = {
                'start_time': env.current_step * env.time_interval,
                'phase': 'turn',
                'target_heading': target_heading
            }

        # 执行Short Skate机动
        skate_state = self.short_skate_states[agent_id]
        current_time = env.current_step * env.time_interval
        elapsed_time = current_time - skate_state['start_time']

        if elapsed_time < 10.0:  # 前10秒执行转弯
            return self._maintain_heading_precise(env, agent_id, skate_state['target_heading'])
        else:  # 10秒后转向返航
            if direction in ["right", "left"]:
                return self._maintain_heading_precise(env, agent_id, 180.0)  # 友方返航南方
            else:
                return self._maintain_heading_precise(env, agent_id, 0.0)    # 敌方返航北方

    # ========== 辅助方法保持不变 ==========

    # ========== 辅助方法 ==========
    def _get_primary_enemy_position(self, env):
        """获取主要敌机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('B') and agent.is_alive:
                return agent.get_position()
        return None

    def _get_primary_friendly_position(self, env):
        """获取主要友机位置"""
        for agent_id, agent in env.agents.items():
            if agent_id.startswith('A') and agent.is_alive:
                return agent.get_position()
        return None

    def _calculate_bearing_to_target(self, my_pos, target_pos):
        """计算指向目标的方位角"""
        # 坐标为NEU: [North, East, Up]
        d_north = target_pos[0] - my_pos[0]
        d_east = target_pos[1] - my_pos[1]
        # 航向以北为0°，顺时针为正，因此使用atan2(East, North)
        bearing = np.rad2deg(np.arctan2(d_east, d_north))
        return bearing % 360

    def _convert_heading_to_index(self, heading_diff_rad):
        """将航向差值转换为指令索引"""
        diff_deg = np.rad2deg(heading_diff_rad)
        heading_options = [-60, -45, -30, -20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 30, 45, 60]

        closest_idx = 0
        min_diff = abs(diff_deg - heading_options[0])

        for i, option in enumerate(heading_options):
            if abs(diff_deg - option) < min_diff:
                min_diff = abs(diff_deg - option)
                closest_idx = i

        return closest_idx

    def _use_lowlevel_policy(self, env, agent_id, altitude_cmd_id, heading_cmd_id, velocity_cmd_id):
        """使用底层策略 - 完全照抄drag_shoot_tactical_task的实现"""
        if self.my_lowlevel_policy is None:
            return np.array([0.0, 0.0, 0.0, 0.7])

        try:
            raw_obs = self.get_obs(env, agent_id)
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
            input_obs = np.expand_dims(input_obs, axis=0)

            if agent_id not in self._inner_rnn_states:
                self._inner_rnn_states[agent_id] = np.zeros((1, 1, 128))

            _action, _rnn_states = self.my_lowlevel_policy(
                torch.FloatTensor(input_obs),
                torch.FloatTensor(self._inner_rnn_states[agent_id])
            )
            action_output = _action.detach().cpu().numpy().squeeze(0)
            self._inner_rnn_states[agent_id] = _rnn_states.detach().cpu().numpy()

            norm_act = np.zeros(4)
            norm_act[0] = action_output[0] / 20 - 1.
            norm_act[1] = action_output[1] / 20 - 1.
            norm_act[2] = action_output[2] / 20 - 1.
            norm_act[3] = action_output[3] / 58 + 0.4

            current_alt = env.agents[agent_id].get_position()[2]
            if current_alt < 1000:
                norm_act[1] = max(norm_act[1], 0.0)
                norm_act[3] = max(norm_act[3], 0.8)

            return norm_act
        except Exception as e:
            logging.error(f"底层策略错误: {e}")
            return np.array([0.0, 0.0, 0.0, 0.7])

    def step(self, env):
        """执行钳形夹击战术步骤 - 基于DragShootTacticalTask的step方法"""
        # 为每个智能体生成战术动作
        obs = {}
        share_obs = {}
        rewards = {}
        dones = {}
        infos = {}

        for agent_id in env._jsbsims.keys():
            if not env._jsbsims[agent_id].is_alive:
                obs[agent_id] = np.zeros(self.obs_length)
                share_obs[agent_id] = np.zeros(self.obs_length)
                rewards[agent_id] = [-10.0]
                dones[agent_id] = [True]
                infos[agent_id] = {"agent_id": agent_id, "alive": False}
                continue

            # 获取观测
            agent_obs = self.get_obs(env, agent_id)
            obs[agent_id] = agent_obs
            share_obs[agent_id] = agent_obs

            # 计算奖励
            rewards[agent_id] = [0.0]  # 简化奖励

            # 检查终止条件
            done, info = self.get_termination(env, agent_id, {})
            dones[agent_id] = [done]
            infos[agent_id] = {"agent_id": agent_id, "alive": True, "phase": self.current_phase.value}

        return obs, share_obs, rewards, dones, infos

    def _handle_missile_launch(self, env, agent_id: str, current_time: float):
        """
        处理导弹发射 - 修复版：更宽松的发射条件，参考拖曳射击项目
        """
        # 检查导弹数量
        if env.agents[agent_id].num_missiles <= 0:
            return

        # 友方返航期间禁止发射导弹
        if agent_id.startswith('A') and self.current_phase == TacticalPhase.DOR_DR:
            logging.info(f"{agent_id} 返航期间禁止发射导弹")
            return

        # 检查冷却时间
        last_launch = self.last_missile_launch_time.get(agent_id, -999)
        if agent_id.startswith('A'):
            cooldown = self.friendly_missile_cooldown
        else:
            cooldown = self.enemy_missile_cooldown

        if current_time - last_launch < cooldown:
            return

        # 寻找目标
        target = None
        min_distance = float('inf')
        for other_id, other_aircraft in env.agents.items():
            if self._is_enemy_agent(agent_id, other_id) and other_aircraft.is_alive:
                distance = self._calculate_distance(env.agents[agent_id], other_aircraft)
                if distance < min_distance:
                    min_distance = distance
                    target = other_aircraft

        if target is None:
            return

        # 修复版导弹发射逻辑：更宽松的条件，敌方也能发射
        should_launch = False

        if agent_id == "A0100":  # 友方长机
            # 更宽松的发射条件：多个阶段都可以发射
            if (self.current_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR, TacticalPhase.TR_DOR] and
                min_distance <= 60000):  # 60km内都可以发射
                should_launch = True
                logging.info(f"🚀 A0100长机发射: 阶段={self.current_phase.value}, 距离={min_distance/1000:.1f}km")

        elif agent_id == "A0200":  # 友方僚机
            # 僚机稍微滞后，但条件也很宽松
            if (self.current_phase in [TacticalPhase.MELD_MTR, TacticalPhase.MTR_TR, TacticalPhase.TR_DOR] and
                min_distance <= 55000):  # 55km内可以发射
                should_launch = True
                logging.info(f"🚀 A0200僚机发射: 阶段={self.current_phase.value}, 距离={min_distance/1000:.1f}km")

        elif agent_id.startswith('B'):  # 敌方
            # shooter-only：只有shooter可以开火（若shooter阵亡会自动移交）
            self._ensure_enemy_roles(env)
            shooter_id = None
            for eid, role in self.enemy_roles.items():
                if role == 'shooter':
                    shooter_id = eid
                    break
            if shooter_id is not None and agent_id != shooter_id:
                should_launch = False
            else:
                # 根据新的对抗阶段逻辑：ENGAGEMENT/ATTACK阶段内发射（提高对抗性）
                if agent_id in self.enemy_combat_states:
                    combat_state = self.enemy_combat_states[agent_id]
                    can_fire = False
                    if (combat_state['phase'] == 'ATTACK' and 40000.0 <= min_distance <= 45000.0):
                        can_fire = True
                    elif (combat_state['phase'] == 'ENGAGEMENT' and 50000.0 <= min_distance <= 60000.0):
                        can_fire = True
                    if can_fire:
                        should_launch = True
                        logging.info(f"🚀 {agent_id} 敌方shooter发射: 阶段={combat_state['phase']}, 距离={min_distance/1000:.1f}km")
                        # 发射后短暂cold
                        combat_state['has_fired'] = True
                        current_time = env.current_step * env.time_interval
                        closest_bearing = 0.0
                        # 计算最近友机方位
                        my_pos = env.agents[agent_id].get_position()
                        min_d = float('inf')
                        for fid in ["A0100", "A0200"]:
                            if fid in env.agents and env.agents[fid].is_alive:
                                d = np.linalg.norm(env.agents[fid].get_position() - my_pos)
                                if d < min_d:
                                    min_d = d
                                    closest_bearing = self._calculate_bearing_to_target(my_pos, env.agents[fid].get_position())
                        side_sign = combat_state.get('side_sign', -1 if self.enemy_roles.get(agent_id,'support')=='shooter' else 1)
                        cold_heading = (closest_bearing + side_sign * 110.0) % 360.0
                        combat_state['hold_heading'] = cold_heading
                        combat_state['hold_until'] = current_time + 6.0
                    else:
                        logging.debug(f"敌方{agent_id}不满足发射条件: 阶段={combat_state['phase']}, 距离={min_distance/1000:.1f}km")

        if should_launch:
            # 执行发射
            if agent_id.startswith('A'):
                self._launch_friendly_missiles(env, agent_id, target, current_time)
            else:
                self._launch_missile(env, agent_id, target, current_time)

    def _is_enemy_agent(self, agent_id1: str, agent_id2: str) -> bool:
        """判断是否为敌方"""
        return (agent_id1.startswith('A') and agent_id2.startswith('B')) or \
               (agent_id1.startswith('B') and agent_id2.startswith('A'))

    def _launch_friendly_missiles(self, env, agent_id: str, target, current_time: float):
        """友方连续发射机制 - 复制拖曳射击项目的实现"""
        aircraft = env.agents[agent_id]
        distance = self._calculate_distance(aircraft, target)

        # 确定发射数量
        missiles_to_launch = 1  # 默认发射1枚

        # 在MTR-TR阶段且距离合适时，考虑发射2枚导弹
        if (self.current_phase == TacticalPhase.MTR_TR and
            aircraft.num_missiles >= 2 and
            41000 <= distance <= 45000):  # MTR-TR阶段最佳发射距离

            # 检查是否已经进行过连续发射
            if not hasattr(self, 'friendly_burst_launch'):
                self.friendly_burst_launch = {"A0100": 0, "A0200": 0}

            burst_count = self.friendly_burst_launch.get(agent_id, 0)
            if burst_count == 0:  # 第一次连续发射机会
                missiles_to_launch = 2
                self.friendly_burst_launch[agent_id] = 1
                logging.info(f"{agent_id} 钳形夹击积极发射策略：一次性发射2枚导弹")

        # 执行发射
        for i in range(missiles_to_launch):
            if aircraft.num_missiles > 0:
                self._launch_missile(env, agent_id, target, current_time)
                if i < missiles_to_launch - 1:  # 不是最后一枚导弹
                    # 短暂延迟，模拟连续发射
                    current_time += 0.5  # 0.5秒间隔

    def _launch_missile(self, env, agent_id: str, target, current_time: float):
        """发射导弹 - 修复版：完全复制拖曳射击项目的导弹创建逻辑"""
        try:
            aircraft = env.agents[agent_id]

            # 创建导弹ID - 使用正确的格式 A0100 → A1001, A1002
            missile_count = 2 - aircraft.num_missiles + 1  # 第1枚或第2枚导弹
            # A0100 → A100, B0100 → B100
            base_id = agent_id[0] + agent_id[2:]  # A0100 → A100
            missile_uid = f"{base_id}{missile_count}"  # A100 → A1001

            # 根据发射平台选择导弹类型 - 完全复制拖曳射击项目
            if agent_id.startswith('A'):  # 我方飞机 - 使用AIM-120C7
                try:
                    from envs.JSBSim.core.simulatior import MissileSimulator
                    missile = MissileSimulator.create(
                        parent=aircraft,
                        target=target,
                        uid=missile_uid
                    )
                    missile_type = "AIM-120C-7"

                    # 添加到环境的临时模拟器 - 关键步骤！
                    env.add_temp_simulator(missile)
                    logging.info(f"✅ 友方导弹{missile_uid}已添加到环境")

                except Exception as missile_error:
                    logging.error(f"友方导弹创建失败: {missile_error}")
                    missile_type = "AIM-120C-7"

            else:  # 敌方飞机 - 使用R-27ER
                try:
                    from r27er_missile import R27ERMissileSimulator
                    missile = R27ERMissileSimulator.create(
                        parent=aircraft,
                        target=target,
                        uid=missile_uid
                    )
                    missile_type = "R-27ER"

                    # 添加到环境的临时模拟器 - 关键步骤！
                    env.add_temp_simulator(missile)
                    logging.info(f"✅ 敌方导弹{missile_uid}已添加到环境")

                except Exception as missile_error:
                    logging.error(f"敌方导弹创建失败: {missile_error}")
                    missile_type = "R-27ER"

            # 初始化导弹记录系统 - 完全复制拖曳射击项目
            if not hasattr(env, '_missile_records'):
                env._missile_records = {}

            # 记录导弹信息
            env._missile_records[missile_uid] = {
                'launcher': agent_id,
                'target': target.uid,
                'type': missile_type,
                'status': 'LAUNCHED',
                'launch_time': current_time,
                'launch_position': aircraft.get_position().copy(),
                'launch_velocity': aircraft.get_velocity().copy()
            }

            # 更新飞机状态
            aircraft.num_missiles -= 1

            # 更新发射时间记录
            self.last_missile_launch_time[agent_id] = current_time

            # 详细的导弹发射日志
            distance_km = self._calculate_distance(aircraft, target)/1000
            logging.info(f"🚀 MISSILE LAUNCH: {agent_id} -> {target.uid} at t={current_time:.1f}s, "
                        f"distance={distance_km:.1f}km, missile_id={missile_uid}, remaining_missiles={aircraft.num_missiles}")

            # 记录发射时间线
            if not hasattr(self, 'missile_launch_timeline'):
                self.missile_launch_timeline = {}
            self.missile_launch_timeline[agent_id] = {
                'launch_time': current_time,
                'phase': self.current_phase.value,
                'distance': distance_km,
                'target': target.uid,
                'missile_type': missile_type,
                'missile_id': missile_uid
            }

            # 发射协调分析
            if agent_id == "A0100":
                logging.info(f"   🎯 长机A0100发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
            elif agent_id == "A0200":
                logging.info(f"   🎯 僚机A0200发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")
                # 检查与长机的协调
                if "A0100" in self.missile_launch_timeline:
                    leader_launch = self.missile_launch_timeline["A0100"]
                    time_diff = current_time - leader_launch['launch_time']
                    logging.info(f"   📊 僚机发射延迟: {time_diff:.1f}s (相对于长机)")
            elif agent_id.startswith("B"):
                logging.info(f"   🎯 敌方{agent_id}发射: {current_time:.1f}s, 阶段={self.current_phase.value}, 距离={distance_km:.1f}km")

            # 打印发射状态汇总
            logging.info(f"📊 导弹发射状态汇总:")
            for launcher, data in self.missile_launch_timeline.items():
                logging.info(f"   {launcher}: {data['launch_time']:.1f}s, {data['phase']}, {data['distance']:.1f}km -> {data['target']}")

        except Exception as e:
            logging.error(f"导弹发射失败 {agent_id}: {e}")
            import traceback
            traceback.print_exc()
